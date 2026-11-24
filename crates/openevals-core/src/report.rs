//! Leaderboard rendering from a results file.

use std::collections::BTreeSet;

use crate::results::ResultsFile;

/// Tasks excluded from the average score because they do not measure quality.
fn is_quality_task(kind: &str) -> bool {
    kind != "efficiency"
}

struct Row<'a> {
    model: &'a str,
    avg: f64,
    scores: Vec<Option<f64>>,
}

fn rows<'a>(results: &'a ResultsFile, tasks: &[String]) -> Vec<Row<'a>> {
    let mut rows: Vec<Row> = results
        .models
        .iter()
        .map(|(model, per_task)| {
            let scores: Vec<Option<f64>> = tasks
                .iter()
                .map(|t| {
                    per_task
                        .get(t)
                        .and_then(|r| r.metrics.get(&r.primary_metric).copied())
                })
                .collect();
            let quality: Vec<f64> = tasks
                .iter()
                .zip(&scores)
                .filter_map(|(t, s)| {
                    let kind = &per_task.get(t)?.kind;
                    if is_quality_task(kind) {
                        *s
                    } else {
                        None
                    }
                })
                .collect();
            let avg = if quality.is_empty() {
                0.0
            } else {
                quality.iter().sum::<f64>() / quality.len() as f64
            };
            Row { model, avg, scores }
        })
        .collect();
    rows.sort_by(|a, b| {
        b.avg
            .partial_cmp(&a.avg)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.model.cmp(b.model))
    });
    rows
}

fn task_names(results: &ResultsFile) -> Vec<String> {
    let set: BTreeSet<String> = results
        .models
        .values()
        .flat_map(|m| m.keys().cloned())
        .collect();
    set.into_iter().collect()
}

pub fn markdown_leaderboard(results: &ResultsFile) -> String {
    let tasks = task_names(results);
    let mut out = String::from("# Benchmark Results\n\n");
    out.push_str("| Rank | Model | Average |");
    for t in &tasks {
        out.push_str(&format!(" {t} |"));
    }
    out.push('\n');
    out.push_str("|---:|:---|---:|");
    for _ in &tasks {
        out.push_str("---:|");
    }
    out.push('\n');
    for (i, r) in rows(results, &tasks).iter().enumerate() {
        out.push_str(&format!("| {} | {} | **{:.4}** |", i + 1, r.model, r.avg));
        for s in &r.scores {
            match s {
                Some(v) => out.push_str(&format!(" {v:.4} |")),
                None => out.push_str(" N/A |"),
            }
        }
        out.push('\n');
    }
    out.push_str("\nScores are the primary metric of each task on a 0 to 1 scale. Efficiency tasks report tokens per second and are excluded from the average.\n");
    out
}

pub fn csv(results: &ResultsFile) -> String {
    let tasks = task_names(results);
    let mut out = String::from("rank,model,average");
    for t in &tasks {
        out.push(',');
        out.push_str(t);
    }
    out.push('\n');
    for (i, r) in rows(results, &tasks).iter().enumerate() {
        out.push_str(&format!("{},{},{:.6}", i + 1, r.model, r.avg));
        for s in &r.scores {
            match s {
                Some(v) => out.push_str(&format!(",{v:.6}")),
                None => out.push(','),
            }
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::TaskResult;
    use std::collections::BTreeMap;

    fn tr(task: &str, kind: &str, metric: &str, v: f64) -> TaskResult {
        TaskResult {
            task: task.into(),
            kind: kind.into(),
            primary_metric: metric.into(),
            metrics: BTreeMap::from([(metric.to_string(), v)]),
            groups: BTreeMap::new(),
            n_items: 10,
            n_errors: 0,
            ci95: None,
            run_stats: BTreeMap::new(),
            config: serde_json::Value::Null,
        }
    }

    #[test]
    fn ranks_by_average_and_skips_efficiency() {
        let mut r = ResultsFile::new();
        r.insert("a", tr("mmlu", "mmlu", "accuracy", 0.5));
        r.insert("b", tr("mmlu", "mmlu", "accuracy", 0.9));
        r.insert("b", tr("eff", "efficiency", "tokens_per_second", 500.0));
        let md = markdown_leaderboard(&r);
        let b_pos = md.find("| 1 | b").expect("b first");
        let a_pos = md.find("| 2 | a").expect("a second");
        assert!(b_pos < a_pos);
        assert!(md.contains("**0.9000**"));
        assert!(md.contains("N/A"));
        assert!(csv(&r).starts_with("rank,model,average,eff,mmlu"));
    }
}
