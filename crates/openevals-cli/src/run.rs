use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use openevals_backend::{BackendConfig, OpenAiBackend};
use openevals_core::cache::Cache;
use openevals_core::config::{ModelEntry, RunConfig};
use openevals_core::registry::{ModelSpec, Reasoning, Registry};
use openevals_core::report;
use openevals_core::results::ResultsFile;
use openevals_tasks::efficiency::run_efficiency;
use openevals_tasks::{build_task, run_task, DataSource, ModelContext, RunOptions};

pub struct Filters {
    pub models: Vec<String>,
    pub tasks: Vec<String>,
    pub output: Option<PathBuf>,
    pub limit: Option<usize>,
    pub dry_run: bool,
}

fn file_safe(label: &str) -> String {
    label
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn backend_for(
    cfg: &RunConfig,
    entry: &ModelEntry,
    spec: Option<&ModelSpec>,
) -> Result<OpenAiBackend> {
    let served = entry
        .served_name
        .clone()
        .or_else(|| spec.map(|s| s.hf_repo.clone()))
        .with_context(|| {
            format!(
                "model '{}' is not in the registry and has no served_name",
                entry.model
            )
        })?;
    let base = entry
        .base_url
        .clone()
        .unwrap_or_else(|| cfg.endpoint.base_url.clone());
    let mut bc = BackendConfig::new(&base, &served);
    bc.label = entry.model.clone();
    bc.timeout = Duration::from_secs(cfg.run.timeout_secs);
    bc.max_retries = cfg.run.retries;
    bc.thinking = entry.thinking;
    bc.reasoning_effort = entry.reasoning_effort.clone();
    let key_env = entry
        .api_key_env
        .clone()
        .or_else(|| cfg.endpoint.api_key_env.clone());
    if let Some(env) = key_env {
        bc.api_key = Some(
            std::env::var(&env)
                .with_context(|| format!("environment variable {env} is not set"))?,
        );
    }
    for (k, v) in &entry.extra_body {
        bc.extra_body.insert(k.clone(), serde_json::to_value(v)?);
    }
    Ok(OpenAiBackend::new(bc)?)
}

fn thinking_on(entry: &ModelEntry, spec: Option<&ModelSpec>) -> bool {
    match entry.thinking {
        Some(t) => t,
        None => spec
            .map(|s| s.reasoning == Reasoning::Always)
            .unwrap_or(false),
    }
}

pub async fn execute(cfg: RunConfig, registry: &Registry, f: Filters) -> Result<()> {
    let entries: Vec<&ModelEntry> = cfg
        .models
        .iter()
        .filter(|m| f.models.is_empty() || f.models.contains(&m.model))
        .collect();
    let task_cfgs: Vec<_> = cfg
        .tasks
        .iter()
        .filter(|t| f.tasks.is_empty() || f.tasks.contains(&t.display_name()))
        .cloned()
        .collect();
    if entries.is_empty() {
        bail!("no models selected");
    }
    if task_cfgs.is_empty() {
        bail!("no tasks selected");
    }
    let output = f
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(&cfg.run.output));

    if f.dry_run {
        println!("output: {}", output.display());
        for e in &entries {
            let spec = registry.get(&e.model);
            let served = e
                .served_name
                .clone()
                .or_else(|| spec.map(|s| s.hf_repo.clone()))
                .unwrap_or_else(|| "?".into());
            println!(
                "model {} -> {} at {}",
                e.model,
                served,
                e.base_url.as_deref().unwrap_or(&cfg.endpoint.base_url)
            );
        }
        for t in &task_cfgs {
            println!("task {} ({})", t.display_name(), t.kind);
        }
        return Ok(());
    }

    let data = DataSource::new(&cfg.run.data_dir);
    let results_path = output.join("results.json");
    let mut results = if results_path.exists() {
        ResultsFile::load(&results_path)?
    } else {
        ResultsFile::new()
    };

    for entry in entries {
        let spec = registry.get(&entry.model);
        let backend = backend_for(&cfg, entry, spec)?;
        let cache = Cache::open(
            &output
                .join("cache")
                .join(format!("{}.jsonl", file_safe(&entry.model))),
        )?;
        let sampling_override = if entry.use_recommended_sampling {
            spec.map(|s| s.recommended_sampling(thinking_on(entry, spec)).clone())
        } else {
            None
        };
        let ctx = ModelContext {
            label: entry.model.clone(),
            sampling_override,
        };
        let opts = RunOptions {
            concurrency: cfg.run.concurrency,
            runs: cfg.run.runs,
            seed: cfg.run.seed,
            samples_dir: Some(output.join("samples")),
        };
        for tc in &task_cfgs {
            let mut tc = tc.clone();
            if let Some(l) = f.limit {
                tc.limit = Some(l);
            }
            let result = if tc.kind == "efficiency" {
                run_efficiency(&backend, &tc).await?
            } else {
                let task = build_task(&tc)?;
                match run_task(task.as_ref(), &data, &backend, &cache, &ctx, &opts).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!(model = %entry.model, task = %tc.display_name(), error = %e, "task failed");
                        continue;
                    }
                }
            };
            println!(
                "{} {} {} = {:.4}",
                entry.model,
                result.task,
                result.primary_metric,
                result
                    .metrics
                    .get(&result.primary_metric)
                    .copied()
                    .unwrap_or(0.0)
            );
            results.insert(&entry.model, result);
            results.save(&results_path)?;
        }
    }
    std::fs::write(
        output.join("leaderboard.md"),
        report::markdown_leaderboard(&results),
    )?;
    println!("results written to {}", output.display());
    Ok(())
}
