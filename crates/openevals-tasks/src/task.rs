use async_trait::async_trait;
use openevals_core::config::TaskConfig;
use openevals_core::{Completion, Error, GenParams, Message, Result};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::data::DataSource;

/// One unit of work: a prompt plus the reference needed for scoring.
#[derive(Clone, Debug)]
pub struct Item {
    pub id: String,
    /// Subject or category used for per-group accuracy.
    pub group: String,
    pub messages: Vec<Message>,
    pub params: GenParams,
    pub reference: Value,
}

#[derive(Clone, Debug, Default)]
pub struct ItemScore {
    pub metrics: BTreeMap<String, f64>,
    pub detail: Value,
}

impl ItemScore {
    pub fn single(name: &str, value: f64, detail: Value) -> Self {
        Self {
            metrics: BTreeMap::from([(name.to_string(), value)]),
            detail,
        }
    }
}

#[async_trait]
pub trait Task: Send + Sync {
    /// Result key for this task.
    fn name(&self) -> &str;
    fn kind(&self) -> &str;
    fn primary_metric(&self) -> &str;
    /// Completions requested per item (pass@k style tasks use more than one).
    fn samples_per_item(&self) -> usize {
        1
    }
    fn limit(&self) -> Option<usize>;
    async fn load(&self, data: &DataSource) -> Result<Vec<Item>>;
    async fn score(&self, item: &Item, completions: &[Completion]) -> ItemScore;
}

pub fn build_task(cfg: &TaskConfig) -> Result<Box<dyn Task>> {
    Ok(match cfg.kind.as_str() {
        "mmlu" | "mmlu_pro" | "arc" | "gpqa" => Box::new(crate::mc::McTask::new(cfg)),
        "truthfulqa" => {
            if cfg.task_type.as_deref() == Some("mc1") {
                Box::new(crate::mc::McTask::new(cfg))
            } else {
                Box::new(crate::truthfulqa::GenerationTask::new(cfg))
            }
        }
        "gsm8k" => Box::new(crate::gsm8k::Gsm8kTask::new(cfg)),
        "math500" => Box::new(crate::math500::Math500Task::new(cfg)),
        "humaneval" => Box::new(crate::humaneval::HumanEvalTask::new(cfg)),
        "efficiency" => {
            return Err(Error::Config(
                "efficiency is run through run_efficiency, not the item runner".into(),
            ))
        }
        other => return Err(Error::Config(format!("unknown task type '{other}'"))),
    })
}

/// Load main rows, honouring a `data_file` override.
pub async fn main_rows(
    cfg: &TaskConfig,
    data: &DataSource,
    dataset: &str,
    config: &str,
    split: &str,
) -> Result<Vec<Value>> {
    match &cfg.data_file {
        Some(p) => crate::data::read_jsonl(std::path::Path::new(p)),
        None => data.rows(dataset, config, split).await,
    }
}
