//! MATH-500 competition math, scored on the final boxed answer.

use async_trait::async_trait;
use openevals_core::config::TaskConfig;
use openevals_core::metrics::{extract_boxed, math_equal};
use openevals_core::{Completion, Error, GenParams, Message, Result};
use serde_json::json;

use crate::data::DataSource;
use crate::task::{main_rows, Item, ItemScore, Task};

pub struct Math500Task {
    cfg: TaskConfig,
    name: String,
}

impl Math500Task {
    pub fn new(cfg: &TaskConfig) -> Self {
        Self {
            name: cfg.display_name(),
            cfg: cfg.clone(),
        }
    }
}

#[async_trait]
impl Task for Math500Task {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> &str {
        "math500"
    }
    fn primary_metric(&self) -> &str {
        "accuracy"
    }
    fn limit(&self) -> Option<usize> {
        self.cfg.limit
    }

    async fn load(&self, data: &DataSource) -> Result<Vec<Item>> {
        let rows = main_rows(&self.cfg, data, "HuggingFaceH4/MATH-500", "default", "test").await?;
        rows.iter()
            .enumerate()
            .map(|(i, r)| {
                let problem = r["problem"]
                    .as_str()
                    .ok_or_else(|| Error::Data(format!("math500 row {i} without problem")))?;
                let answer = r["answer"]
                    .as_str()
                    .ok_or_else(|| Error::Data(format!("math500 row {i} without answer")))?;
                Ok(Item {
                    id: format!("math500-{i}"),
                    group: r["subject"].as_str().unwrap_or("all").to_string(),
                    messages: vec![Message::user(format!(
                        "{problem}\n\nReason step by step and put the final answer in \\boxed{{}}."
                    ))],
                    params: GenParams {
                        temperature: self.cfg.temperature.unwrap_or(0.0),
                        max_tokens: self.cfg.max_tokens.unwrap_or(2048),
                        ..Default::default()
                    },
                    reference: json!({"answer": answer}),
                })
            })
            .collect()
    }

    async fn score(&self, item: &Item, completions: &[Completion]) -> ItemScore {
        let truth = item.reference["answer"].as_str().unwrap_or("");
        let pred = completions.first().and_then(|c| extract_boxed(&c.text));
        let ok = pred
            .as_deref()
            .map(|p| math_equal(p, truth))
            .unwrap_or(false);
        ItemScore::single(
            "accuracy",
            if ok { 1.0 } else { 0.0 },
            json!({"predicted": pred, "truth": truth}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn scores_boxed_answers() {
        let t = Math500Task::new(&TaskConfig {
            kind: "math500".into(),
            ..Default::default()
        });
        let item = Item {
            id: "1".into(),
            group: "Algebra".into(),
            messages: vec![],
            params: GenParams::default(),
            reference: json!({"answer": "\\frac{1}{2}"}),
        };
        let c = |s: &str| {
            vec![Completion {
                text: s.into(),
                ..Default::default()
            }]
        };
        assert_eq!(
            t.score(&item, &c("so \\boxed{\\dfrac{1}{2}}"))
                .await
                .metrics["accuracy"],
            1.0
        );
        assert_eq!(t.score(&item, &c("so 1/2")).await.metrics["accuracy"], 0.0);
        assert_eq!(
            t.score(&item, &c("\\boxed{3}")).await.metrics["accuracy"],
            0.0
        );
    }
}
