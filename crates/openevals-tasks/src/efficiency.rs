//! Latency and throughput measurement against a live endpoint.
//!
//! Memory, GPU and CPU utilisation from the Python version are not measured
//! here because the model runs in a separate server process.

use std::collections::BTreeMap;

use openevals_core::config::TaskConfig;
use openevals_core::results::TaskResult;
use openevals_core::{GenParams, GenRequest, Generator, Message, Result};
use serde_json::json;

const DEFAULT_PROMPTS: &[&str] = &[
    "Explain the theory of relativity in simple terms",
    "Write a short story about a robot who discovers emotions",
    "Summarize the key events of World War II",
    "Describe the process of photosynthesis in plants",
    "What are the benefits of renewable energy?",
];

/// Tokens per second over the whole request. Returns None without tokens.
pub fn tokens_per_second(completion_tokens: u64, latency_ms: f64) -> Option<f64> {
    if completion_tokens == 0 || latency_ms <= 0.0 {
        None
    } else {
        Some(completion_tokens as f64 / (latency_ms / 1000.0))
    }
}

fn mean(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        None
    } else {
        Some(v.iter().sum::<f64>() / v.len() as f64)
    }
}

pub async fn run_efficiency(gen: &dyn Generator, cfg: &TaskConfig) -> Result<TaskResult> {
    let prompts: Vec<String> = cfg
        .sample_prompts
        .clone()
        .unwrap_or_else(|| DEFAULT_PROMPTS.iter().map(|s| s.to_string()).collect());
    let lengths = cfg
        .output_lengths
        .clone()
        .unwrap_or_else(|| vec![64, 128, 256, 512]);
    let warmup = cfg.warmup_runs.unwrap_or(2);
    let runs = cfg.measurement_runs.unwrap_or(3);

    let request = |prompt: &str, max_tokens: u32| GenRequest {
        messages: vec![Message::user(prompt)],
        params: GenParams {
            temperature: 0.0,
            max_tokens,
            ..Default::default()
        },
    };

    for i in 0..warmup {
        let p = &prompts[i % prompts.len().max(1)];
        let _ = gen.generate_timed(&request(p, 16)).await;
    }

    let mut metrics = BTreeMap::new();
    let mut all_tps = Vec::new();
    let mut n_errors = 0usize;
    let mut n_ok = 0usize;
    for len in &lengths {
        let (mut lat, mut ttft, mut tps) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..runs {
            for p in &prompts {
                match gen.generate_timed(&request(p, *len)).await {
                    Ok(c) => {
                        n_ok += 1;
                        lat.push(c.latency_ms);
                        if let Some(t) = c.ttft_ms {
                            ttft.push(t);
                        }
                        if let Some(v) =
                            tokens_per_second(c.completion_tokens.unwrap_or(0), c.latency_ms)
                        {
                            tps.push(v);
                        }
                    }
                    Err(e) => {
                        n_errors += 1;
                        tracing::warn!(error = %e, "efficiency measurement failed");
                    }
                }
            }
        }
        if let Some(v) = mean(&lat) {
            metrics.insert(format!("tokens_{len}.latency_ms"), v);
        }
        if let Some(v) = mean(&ttft) {
            metrics.insert(format!("tokens_{len}.ttft_ms"), v);
        }
        if let Some(v) = mean(&tps) {
            metrics.insert(format!("tokens_{len}.tokens_per_second"), v);
            all_tps.push(v);
        }
    }
    if let Some(v) = mean(&all_tps) {
        metrics.insert("tokens_per_second".into(), v);
    }
    Ok(TaskResult {
        task: cfg.display_name(),
        kind: "efficiency".into(),
        primary_metric: "tokens_per_second".into(),
        metrics,
        groups: BTreeMap::new(),
        n_items: n_ok,
        n_errors,
        ci95: None,
        run_stats: BTreeMap::new(),
        config: json!({
            "output_lengths": lengths,
            "warmup_runs": warmup,
            "measurement_runs": runs,
            "num_prompts": prompts.len(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_throughput() {
        assert_eq!(tokens_per_second(100, 2000.0), Some(50.0));
        assert_eq!(tokens_per_second(0, 2000.0), None);
        assert_eq!(tokens_per_second(10, 0.0), None);
    }
}
