//! Run configuration (TOML).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

pub const TASK_TYPES: &[&str] = &[
    "mmlu",
    "mmlu_pro",
    "gsm8k",
    "math500",
    "humaneval",
    "arc",
    "truthfulqa",
    "gpqa",
    "efficiency",
];

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    #[serde(default)]
    pub run: RunSettings,
    #[serde(default)]
    pub endpoint: EndpointConfig,
    #[serde(default, rename = "models")]
    pub models: Vec<ModelEntry>,
    #[serde(default, rename = "tasks")]
    pub tasks: Vec<TaskConfig>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct RunSettings {
    /// Output directory for results, samples and the completion cache.
    pub output: String,
    /// Directory where dataset rows are cached as JSONL.
    pub data_dir: String,
    /// Maximum in-flight requests per model.
    pub concurrency: usize,
    /// Retries after the first failed attempt.
    pub retries: u32,
    /// Per-request timeout in seconds.
    pub timeout_secs: u64,
    /// Number of repeated runs for mean and standard deviation.
    pub runs: u32,
    /// Base seed for shuffles and request seeds.
    pub seed: u64,
}

impl Default for RunSettings {
    fn default() -> Self {
        Self {
            output: "results".into(),
            data_dir: "data".into(),
            concurrency: 8,
            retries: 5,
            timeout_secs: 600,
            runs: 1,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct EndpointConfig {
    /// Base URL of an OpenAI-compatible server, including the `/v1` prefix.
    pub base_url: String,
    /// Name of the environment variable holding the API key, if any.
    pub api_key_env: Option<String>,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8000/v1".into(),
            api_key_env: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    /// Registry id, or any label when `served_name` is given.
    pub model: String,
    /// Model name sent to the server. Defaults to the registry `hf_repo`.
    pub served_name: Option<String>,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    /// Toggle thinking via `chat_template_kwargs.enable_thinking`.
    pub thinking: Option<bool>,
    /// Sent as `reasoning_effort` (for example for gpt-oss).
    pub reasoning_effort: Option<String>,
    /// Use the vendor recommended sampling from the registry instead of greedy.
    #[serde(default)]
    pub use_recommended_sampling: bool,
    /// Extra JSON fields merged into every request body.
    #[serde(default)]
    pub extra_body: BTreeMap<String, toml::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct TaskConfig {
    /// Unique name used in result files. Defaults to `type`.
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub kind: String,
    pub subset: Option<String>,
    pub shots: Option<usize>,
    /// Evaluate only the first N items.
    pub limit: Option<usize>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    /// HumanEval: samples per problem.
    pub samples: Option<usize>,
    /// HumanEval: values of k for pass@k.
    pub k_values: Option<Vec<usize>>,
    /// HumanEval: seconds allowed per test program.
    pub timeout: Option<u64>,
    /// HumanEval: python interpreter.
    pub python: Option<String>,
    /// TruthfulQA: `generation` or `mc1`.
    pub task_type: Option<String>,
    /// Efficiency: prompts, output lengths, warmup and measurement runs.
    pub sample_prompts: Option<Vec<String>>,
    pub output_lengths: Option<Vec<u32>>,
    pub warmup_runs: Option<usize>,
    pub measurement_runs: Option<usize>,
    /// Override dataset location: a JSONL file with rows in the dataset schema.
    pub data_file: Option<String>,
}

impl TaskConfig {
    pub fn display_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| self.kind.clone())
    }
}

impl RunConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("cannot read {}: {e}", path.display())))?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let cfg: RunConfig = toml::from_str(text).map_err(|e| Error::Config(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::Config(m));
        if self.run.concurrency == 0 {
            return bad("run.concurrency must be at least 1".into());
        }
        if self.run.runs == 0 {
            return bad("run.runs must be at least 1".into());
        }
        if self.run.timeout_secs == 0 {
            return bad("run.timeout_secs must be at least 1".into());
        }
        if !self.endpoint.base_url.starts_with("http") {
            return bad(format!(
                "endpoint.base_url must start with http, got '{}'",
                self.endpoint.base_url
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        for t in &self.tasks {
            if !TASK_TYPES.contains(&t.kind.as_str()) {
                return bad(format!(
                    "unknown task type '{}' (supported: {})",
                    t.kind,
                    TASK_TYPES.join(", ")
                ));
            }
            if !names.insert(t.display_name()) {
                return bad(format!("duplicate task name '{}'", t.display_name()));
            }
            if let Some(s) = t.shots {
                let max = if t.kind == "gsm8k" { 8 } else { 25 };
                if s > max {
                    return bad(format!(
                        "task '{}': shots must be at most {max}",
                        t.display_name()
                    ));
                }
            }
            if let Some(tt) = &t.task_type {
                if t.kind == "truthfulqa" && tt != "generation" && tt != "mc1" {
                    return bad(format!(
                        "truthfulqa task_type must be generation or mc1, got '{tt}'"
                    ));
                }
            }
            if let Some(t0) = t.temperature {
                if !(0.0..=2.0).contains(&t0) {
                    return bad(format!(
                        "task '{}': temperature must be in [0, 2]",
                        t.display_name()
                    ));
                }
            }
            if t.samples == Some(0) || t.max_tokens == Some(0) {
                return bad(format!(
                    "task '{}': samples and max_tokens must be positive",
                    t.display_name()
                ));
            }
        }
        let mut labels = std::collections::BTreeSet::new();
        for m in &self.models {
            if !labels.insert(m.model.clone()) {
                return bad(format!("duplicate model entry '{}'", m.model));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"
[run]
output = "out"
concurrency = 4

[endpoint]
base_url = "http://localhost:11434/v1"

[[models]]
model = "qwen3-8b"
thinking = false

[[tasks]]
type = "mmlu"
shots = 5
limit = 10

[[tasks]]
type = "gsm8k"
name = "gsm8k-0shot"
shots = 0
"#;

    #[test]
    fn parses_valid_config() {
        let c = RunConfig::parse(GOOD).unwrap();
        assert_eq!(c.run.concurrency, 4);
        assert_eq!(c.run.retries, 5);
        assert_eq!(c.models[0].thinking, Some(false));
        assert_eq!(c.tasks[1].display_name(), "gsm8k-0shot");
    }

    #[test]
    fn rejects_unknown_task_and_fields() {
        assert!(RunConfig::parse("[[tasks]]\ntype='nope'").is_err());
        assert!(RunConfig::parse("[run]\nbogus=1").is_err());
        assert!(RunConfig::parse("[run]\nconcurrency=0").is_err());
        assert!(RunConfig::parse("[[tasks]]\ntype='gsm8k'\nshots=9").is_err());
        assert!(RunConfig::parse("[[tasks]]\ntype='mmlu'\n[[tasks]]\ntype='mmlu'").is_err());
        assert!(RunConfig::parse("[[tasks]]\ntype='truthfulqa'\ntask_type='mc2'").is_err());
    }
}
