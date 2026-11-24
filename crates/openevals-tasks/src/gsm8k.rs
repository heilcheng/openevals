//! GSM8K grade school math.

use async_trait::async_trait;
use openevals_core::config::TaskConfig;
use openevals_core::metrics::{extract_numerical_answer, is_exact_match};
use openevals_core::{Completion, Error, GenParams, Message, Result};
use serde_json::json;

use crate::data::DataSource;
use crate::task::{main_rows, Item, ItemScore, Task};

pub struct Gsm8kTask {
    cfg: TaskConfig,
    name: String,
}

impl Gsm8kTask {
    pub fn new(cfg: &TaskConfig) -> Self {
        Self {
            name: cfg.display_name(),
            cfg: cfg.clone(),
        }
    }
}

const INSTRUCTION: &str = "Solve the grade school math problem. Reason step by step, then give the final numeric answer on its own line in the form \"#### <number>\".";

pub fn format_example(question: &str, answer: Option<&str>) -> String {
    match answer {
        Some(a) => format!(
            "**Question:**\n{}\n**Answer:**\n{}",
            question.trim(),
            a.trim()
        ),
        None => format!("**Question:**\n{}\n**Answer:**\n", question.trim()),
    }
}

/// Ground truth is the number after the `####` marker.
pub fn ground_truth(answer: &str) -> Option<f64> {
    let tail = answer.rsplit("####").next()?;
    if tail.len() == answer.len() {
        return None;
    }
    extract_numerical_answer(&format!("#### {}", tail.trim()))
}

#[async_trait]
impl Task for Gsm8kTask {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> &str {
        "gsm8k"
    }
    fn primary_metric(&self) -> &str {
        "accuracy"
    }
    fn limit(&self) -> Option<usize> {
        self.cfg.limit
    }

    async fn load(&self, data: &DataSource) -> Result<Vec<Item>> {
        let shots_n = self.cfg.shots.unwrap_or(8);
        let test = main_rows(&self.cfg, data, "openai/gsm8k", "main", "test").await?;
        let train = if shots_n > 0 {
            data.rows("openai/gsm8k", "main", "train").await?
        } else {
            Vec::new()
        };
        // Fixed shots from the head of the train split keep prompts identical
        // across items and runs.
        let shots: Vec<String> = train
            .iter()
            .take(shots_n)
            .map(|r| format_example(r["question"].as_str().unwrap_or(""), r["answer"].as_str()))
            .collect();
        let mut items = Vec::new();
        for (i, r) in test.iter().enumerate() {
            let question = r["question"]
                .as_str()
                .ok_or_else(|| Error::Data(format!("gsm8k row {i} without question")))?;
            let answer = r["answer"]
                .as_str()
                .ok_or_else(|| Error::Data(format!("gsm8k row {i} without answer")))?;
            let mut prompt = String::from(INSTRUCTION);
            prompt.push_str("\n\n");
            for s in &shots {
                prompt.push_str(s);
                prompt.push_str("\n\n");
            }
            prompt.push_str(&format_example(question, None));
            items.push(Item {
                id: format!("gsm8k-{i}"),
                group: "all".into(),
                messages: vec![Message::user(prompt)],
                params: GenParams {
                    temperature: self.cfg.temperature.unwrap_or(0.0),
                    max_tokens: self.cfg.max_tokens.unwrap_or(1024),
                    ..Default::default()
                },
                reference: json!({ "truth": ground_truth(answer) }),
            });
        }
        Ok(items)
    }

    async fn score(&self, item: &Item, completions: &[Completion]) -> ItemScore {
        let truth = item.reference["truth"].as_f64();
        let pred = completions
            .first()
            .and_then(|c| extract_numerical_answer(&c.text));
        let ok = is_exact_match(pred, truth);
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

    #[test]
    fn parses_ground_truth() {
        assert_eq!(
            ground_truth("Natalia sold 48+24 = <<48+24=72>>72\n#### 72"),
            Some(72.0)
        );
        assert_eq!(ground_truth("x\n#### 1,200"), Some(1200.0));
        assert_eq!(ground_truth("no marker"), None);
    }

    #[tokio::test]
    async fn scores_items() {
        let task = Gsm8kTask::new(&TaskConfig {
            kind: "gsm8k".into(),
            ..Default::default()
        });
        let item = Item {
            id: "1".into(),
            group: "all".into(),
            messages: vec![],
            params: GenParams::default(),
            reference: json!({"truth": 72.0}),
        };
        let good = Completion {
            text: "so\n#### 72".into(),
            ..Default::default()
        };
        let bad = Completion {
            text: "#### 73".into(),
            ..Default::default()
        };
        assert_eq!(task.score(&item, &[good]).await.metrics["accuracy"], 1.0);
        assert_eq!(task.score(&item, &[bad]).await.metrics["accuracy"], 0.0);
        assert_eq!(task.score(&item, &[]).await.metrics["accuracy"], 0.0);
    }
}
