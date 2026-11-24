//! Runs one task against one model: concurrent generation with a resumable
//! completion cache, deterministic scoring and aggregation.

use std::collections::BTreeMap;
use std::path::PathBuf;

use futures::stream::{self, StreamExt};
use openevals_core::cache::{request_key, Cache};
use openevals_core::metrics::{mean_std, wilson_ci};
use openevals_core::registry::Sampling;
use openevals_core::results::{MetricStats, TaskResult};
use openevals_core::{Completion, GenParams, GenRequest, Generator, Result};
use serde_json::{json, Value};

use crate::data::DataSource;
use crate::task::{Item, ItemScore, Task};

/// Per-model settings that affect requests.
#[derive(Clone, Default)]
pub struct ModelContext {
    /// Label used in file names and cache keys.
    pub label: String,
    /// Sampling that replaces the task defaults when set.
    pub sampling_override: Option<Sampling>,
}

#[derive(Clone)]
pub struct RunOptions {
    pub concurrency: usize,
    pub runs: u32,
    pub seed: u64,
    /// When set, per-sample records are written under this directory.
    pub samples_dir: Option<PathBuf>,
}

fn apply_override(params: &mut GenParams, s: &Sampling) {
    if let Some(t) = s.temperature {
        params.temperature = t;
    }
    if s.top_p.is_some() {
        params.top_p = s.top_p;
    }
    if s.top_k.is_some() {
        params.top_k = s.top_k;
    }
    if s.min_p.is_some() {
        params.min_p = s.min_p;
    }
}

async fn generate_cached(
    gen: &dyn Generator,
    cache: &Cache,
    model_key: &str,
    req: &GenRequest,
    run: u32,
) -> Completion {
    let key = request_key(model_key, req, run);
    if let Some(c) = cache.get(&key) {
        return c;
    }
    match gen.generate(req).await {
        Ok(c) => {
            if let Err(e) = cache.put(&key, &c) {
                tracing::warn!(error = %e, "failed to write cache entry");
            }
            c
        }
        Err(e) => Completion {
            error: Some(e.to_string()),
            ..Default::default()
        },
    }
}

struct Scored {
    item: Item,
    completions: Vec<Completion>,
    score: ItemScore,
}

async fn run_once(
    task: &dyn Task,
    items: &[Item],
    gen: &dyn Generator,
    cache: &Cache,
    ctx: &ModelContext,
    opts: &RunOptions,
    run: u32,
) -> Vec<Scored> {
    let samples = task.samples_per_item();
    let total = items.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let jobs = items.iter().map(|item| {
        let done = &done;
        async move {
            let mut params = item.params.clone();
            if let Some(s) = &ctx.sampling_override {
                apply_override(&mut params, s);
            }
            let mut completions = Vec::with_capacity(samples);
            for s in 0..samples {
                let mut p = params.clone();
                if samples > 1 || run > 0 {
                    p.seed = Some(opts.seed + s as u64 + 1000 * run as u64);
                }
                let req = GenRequest {
                    messages: item.messages.clone(),
                    params: p,
                };
                completions.push(generate_cached(gen, cache, &ctx.label, &req, run).await);
            }
            let score = if completions.iter().any(|c| c.error.is_some()) {
                ItemScore {
                    metrics: BTreeMap::new(),
                    detail: json!({"error": completions.iter().find_map(|c| c.error.clone())}),
                }
            } else {
                task.score(item, &completions).await
            };
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_multiple_of(25) || n == total {
                tracing::info!(task = task.name(), "{n}/{total} items");
            }
            Scored {
                item: item.clone(),
                completions,
                score,
            }
        }
    });
    stream::iter(jobs)
        .buffered(opts.concurrency.max(1))
        .collect()
        .await
}

/// Aggregate item scores into a task result for one run.
#[derive(Clone, Default)]
struct Aggregate {
    metrics: BTreeMap<String, f64>,
    groups: BTreeMap<String, f64>,
    n_errors: usize,
    ci: Option<(f64, f64)>,
}

fn aggregate(task: &dyn Task, scored: &[Scored]) -> Aggregate {
    let n_items = scored.len();
    let n_errors = scored.iter().filter(|s| s.score.metrics.is_empty()).count();
    // Errored items count as zero for every metric the task produces.
    let metric_names: Vec<String> = scored
        .iter()
        .flat_map(|s| s.score.metrics.keys().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let value = |s: &Scored, m: &str| s.score.metrics.get(m).copied().unwrap_or(0.0);
    let mut metrics = BTreeMap::new();
    for m in &metric_names {
        if n_items > 0 {
            metrics.insert(
                m.clone(),
                scored.iter().map(|s| value(s, m)).sum::<f64>() / n_items as f64,
            );
        }
    }
    let primary = task.primary_metric();
    let mut sums: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for s in scored {
        let e = sums.entry(s.item.group.clone()).or_default();
        e.0 += value(s, primary);
        e.1 += 1;
    }
    let groups = if sums.len() > 1 {
        sums.into_iter()
            .map(|(g, (sum, n))| (g, sum / n as f64))
            .collect()
    } else {
        BTreeMap::new()
    };
    let binary = !scored.is_empty()
        && scored
            .iter()
            .all(|s| matches!(value(s, primary), v if v == 0.0 || v == 1.0));
    let ci = if binary && task.samples_per_item() == 1 {
        let succ: f64 = scored.iter().map(|s| value(s, primary)).sum();
        wilson_ci(succ, n_items)
    } else {
        None
    };
    Aggregate {
        metrics,
        groups,
        n_errors,
        ci,
    }
}

fn write_samples(
    dir: &std::path::Path,
    ctx: &ModelContext,
    task: &dyn Task,
    scored: &[Scored],
) -> Result<()> {
    let path = dir.join(&ctx.label).join(format!("{}.jsonl", task.name()));
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let mut out = String::new();
    for s in scored {
        let rec: Value = json!({
            "id": s.item.id,
            "group": s.item.group,
            "prompt": s.item.messages.last().map(|m| m.content.clone()),
            "responses": s.completions.iter().map(|c| c.text.clone()).collect::<Vec<_>>(),
            "metrics": s.score.metrics,
            "detail": s.score.detail,
            "completion_tokens": s.completions.iter().map(|c| c.completion_tokens).collect::<Vec<_>>(),
            "finish_reason": s.completions.iter().map(|c| c.finish_reason.clone()).collect::<Vec<_>>(),
        });
        out.push_str(&serde_json::to_string(&rec)?);
        out.push('\n');
    }
    std::fs::write(path, out)?;
    Ok(())
}

pub async fn run_task(
    task: &dyn Task,
    data: &DataSource,
    gen: &dyn Generator,
    cache: &Cache,
    ctx: &ModelContext,
    opts: &RunOptions,
) -> Result<TaskResult> {
    let mut items = task.load(data).await?;
    if let Some(l) = task.limit() {
        items.truncate(l);
    }
    tracing::info!(task = task.name(), model = %ctx.label, items = items.len(), "starting");

    let mut per_run = Vec::new();
    let mut last_scored = Vec::new();
    for run in 0..opts.runs.max(1) {
        let scored = run_once(task, &items, gen, cache, ctx, opts, run).await;
        per_run.push(aggregate(task, &scored));
        last_scored = scored;
    }
    if let Some(dir) = &opts.samples_dir {
        write_samples(dir, ctx, task, &last_scored)?;
    }

    let n_runs = per_run.len() as f64;
    let mut metrics: BTreeMap<String, f64> = BTreeMap::new();
    let mut run_stats = BTreeMap::new();
    let names: std::collections::BTreeSet<String> = per_run
        .iter()
        .flat_map(|r| r.metrics.keys().cloned())
        .collect();
    for m in names {
        let values: Vec<f64> = per_run
            .iter()
            .map(|r| r.metrics.get(&m).copied().unwrap_or(0.0))
            .collect();
        metrics.insert(m.clone(), values.iter().sum::<f64>() / n_runs);
        if per_run.len() > 1 {
            if let Some(s) = mean_std(&values) {
                run_stats.insert(m, MetricStats::from_stats(&s, values));
            }
        }
    }
    let last = per_run.last().cloned().unwrap_or_default();
    let (groups, n_errors, ci95) = (last.groups, last.n_errors, last.ci);
    Ok(TaskResult {
        task: task.name().to_string(),
        kind: task.kind().to_string(),
        primary_metric: task.primary_metric().to_string(),
        metrics,
        groups,
        n_items: items.len(),
        n_errors,
        ci95: if per_run.len() == 1 { ci95 } else { None },
        run_stats,
        config: json!({"runs": opts.runs, "samples_per_item": task.samples_per_item()}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use openevals_core::{Error, Message};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Echo {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Generator for Echo {
        fn model_id(&self) -> &str {
            "echo"
        }
        async fn generate(&self, req: &GenRequest) -> Result<Completion> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let q = &req.messages[0].content;
            if q.contains("boom") {
                return Err(Error::Http("down".into()));
            }
            // answers B for q1 (correct) and A otherwise
            let text = if q.contains("q1") {
                "Answer: B"
            } else {
                "Answer: A"
            };
            Ok(Completion {
                text: text.into(),
                completion_tokens: Some(3),
                ..Default::default()
            })
        }
    }

    struct Fixed(Vec<(&'static str, &'static str)>);

    #[async_trait]
    impl Task for Fixed {
        fn name(&self) -> &str {
            "fixed"
        }
        fn kind(&self) -> &str {
            "mmlu"
        }
        fn primary_metric(&self) -> &str {
            "accuracy"
        }
        fn limit(&self) -> Option<usize> {
            None
        }
        async fn load(&self, _: &DataSource) -> Result<Vec<Item>> {
            Ok(self
                .0
                .iter()
                .map(|(q, g)| Item {
                    id: q.to_string(),
                    group: g.to_string(),
                    messages: vec![Message::user(*q)],
                    params: GenParams::default(),
                    reference: json!({"answer": 1, "n": 4}),
                })
                .collect())
        }
        async fn score(&self, item: &Item, c: &[Completion]) -> ItemScore {
            let ok = c[0].text.ends_with('B') && item.id.contains("q1") || item.id.contains("q3");
            ItemScore::single("accuracy", if ok { 1.0 } else { 0.0 }, json!({}))
        }
    }

    fn opts() -> RunOptions {
        RunOptions {
            concurrency: 2,
            runs: 1,
            seed: 1,
            samples_dir: None,
        }
    }

    #[tokio::test]
    async fn aggregates_groups_errors_and_resumes_from_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(&dir.path().join("c.jsonl")).unwrap();
        let data = DataSource::new(dir.path());
        let task = Fixed(vec![("q1", "x"), ("q2", "x"), ("boom", "y"), ("q3", "y")]);
        let gen = Echo {
            calls: AtomicUsize::new(0),
        };
        let ctx = ModelContext {
            label: "echo".into(),
            sampling_override: None,
        };
        let r = run_task(&task, &data, &gen, &cache, &ctx, &opts())
            .await
            .unwrap();
        assert_eq!(r.n_items, 4);
        assert_eq!(r.n_errors, 1);
        // q1 correct, q3 correct by construction, q2 wrong, boom errored (0)
        assert!((r.metrics["accuracy"] - 0.5).abs() < 1e-9);
        assert!((r.groups["x"] - 0.5).abs() < 1e-9);
        assert!((r.groups["y"] - 0.5).abs() < 1e-9);
        assert!(r.ci95.is_some());
        let first_calls = gen.calls.load(Ordering::SeqCst);
        assert_eq!(first_calls, 4);

        // second run: successful requests come from the cache, only the failed one is retried
        let r2 = run_task(&task, &data, &gen, &cache, &ctx, &opts())
            .await
            .unwrap();
        assert_eq!(gen.calls.load(Ordering::SeqCst), first_calls + 1);
        assert_eq!(r.metrics, r2.metrics);
    }

    #[tokio::test]
    async fn multiple_runs_report_stats() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(&dir.path().join("c.jsonl")).unwrap();
        let data = DataSource::new(dir.path());
        let task = Fixed(vec![("q1", "x"), ("q2", "x")]);
        let gen = Echo {
            calls: AtomicUsize::new(0),
        };
        let ctx = ModelContext {
            label: "echo".into(),
            sampling_override: None,
        };
        let mut o = opts();
        o.runs = 3;
        o.samples_dir = Some(dir.path().join("samples"));
        let r = run_task(&task, &data, &gen, &cache, &ctx, &o)
            .await
            .unwrap();
        let st = &r.run_stats["accuracy"];
        assert_eq!(st.values.len(), 3);
        assert_eq!(st.std, 0.0);
        assert!(r.ci95.is_none());
        assert!(dir.path().join("samples/echo/fixed.jsonl").exists());
        // distinct run indices produce distinct cache keys
        assert_eq!(gen.calls.load(Ordering::SeqCst), 6);
    }
}
