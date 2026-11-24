//! Result file format (`results.json`).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::metrics::Stats;
use crate::Result;

pub const FORMAT_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MetricStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub values: Vec<f64>,
}

impl MetricStats {
    pub fn from_stats(s: &Stats, values: Vec<f64>) -> Self {
        Self {
            mean: s.mean,
            std: s.std,
            min: s.min,
            max: s.max,
            values,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TaskResult {
    pub task: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub primary_metric: String,
    /// Mean of each metric over items (and over runs when runs > 1).
    pub metrics: BTreeMap<String, f64>,
    /// Mean of the primary metric per subject or category.
    #[serde(default)]
    pub groups: BTreeMap<String, f64>,
    pub n_items: usize,
    /// Items for which generation failed after all retries.
    #[serde(default)]
    pub n_errors: usize,
    /// 95 percent Wilson interval of the primary metric when it is binary.
    #[serde(default)]
    pub ci95: Option<(f64, f64)>,
    /// Per-metric statistics across runs, present when runs > 1.
    #[serde(default)]
    pub run_stats: BTreeMap<String, MetricStats>,
    #[serde(default)]
    pub config: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ResultsFile {
    pub version: u32,
    /// model label to task name to result
    pub models: BTreeMap<String, BTreeMap<String, TaskResult>>,
}

impl ResultsFile {
    pub fn new() -> Self {
        Self {
            version: FORMAT_VERSION,
            models: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, model: &str, result: TaskResult) {
        self.models
            .entry(model.to_string())
            .or_default()
            .insert(result.task.clone(), result);
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }
}
