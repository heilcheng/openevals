//! Multiple choice benchmarks: MMLU, MMLU-Pro, ARC, GPQA and TruthfulQA MC1.

use async_trait::async_trait;
use openevals_core::config::TaskConfig;
use openevals_core::rng::{seed_for, SplitMix64};
use openevals_core::{Completion, Error, GenParams, Message, Result};
use regex::Regex;
use serde_json::{json, Value};
use std::sync::OnceLock;

use crate::data::DataSource;
use crate::task::{main_rows, Item, ItemScore, Task};

const LETTERS: &[u8] = b"ABCDEFGHIJKLMNOP";

#[derive(Clone, Debug, PartialEq)]
pub struct McQuestion {
    pub id: String,
    pub group: String,
    pub question: String,
    pub options: Vec<String>,
    pub answer: usize,
}

pub fn render_question(q: &str, options: &[String], answer: Option<usize>) -> String {
    let mut s = format!("{}\n", q.trim());
    for (i, o) in options.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", LETTERS[i] as char, o.trim()));
    }
    match answer {
        Some(a) => s.push_str(&format!("Answer: {}", LETTERS[a] as char)),
        None => s.push_str("Answer:"),
    }
    s
}

/// Build the user prompt: instruction, optional few-shot examples, question.
pub fn build_prompt(subject: &str, shots: &[McQuestion], q: &McQuestion) -> String {
    let mut s = format!(
        "The following are multiple choice questions (with answers) about {subject}. \
Answer the final question. End your reply with a line of the form \"Answer: X\" where X is the letter of the correct option.\n\n"
    );
    for sh in shots {
        s.push_str(&render_question(&sh.question, &sh.options, Some(sh.answer)));
        s.push_str("\n\n");
    }
    s.push_str(&render_question(&q.question, &q.options, None));
    s
}

/// Extract the chosen option index from a reply.
///
/// Accepts the last "Answer: X" style statement, or a reply that begins with
/// a bare option letter. Anything else counts as no answer.
pub fn extract_choice(text: &str, n_options: usize) -> Option<usize> {
    static STATED: OnceLock<Regex> = OnceLock::new();
    static LEADING: OnceLock<Regex> = OnceLock::new();
    let stated = STATED.get_or_init(|| {
        Regex::new(r"(?i:answer)\s*(?:is|:)?\s*[:\-]?\s*[\*\(\[]*\s*([A-P])\b").unwrap()
    });
    let leading = LEADING
        .get_or_init(|| Regex::new(r"^\s*[\*\(\[]*([A-P])(?:[\*\)\]\.:]+(?:\s|$)|\s*$)").unwrap());
    let idx = |c: &str| (c.as_bytes()[0] - b'A') as usize;
    let candidate = stated
        .captures_iter(text)
        .last()
        .map(|c| idx(&c[1]))
        .or_else(|| leading.captures(text).map(|c| idx(&c[1])));
    candidate.filter(|i| *i < n_options)
}

pub struct McTask {
    cfg: TaskConfig,
    name: String,
}

impl McTask {
    pub fn new(cfg: &TaskConfig) -> Self {
        Self {
            name: cfg.display_name(),
            cfg: cfg.clone(),
        }
    }

    fn params(&self) -> GenParams {
        GenParams {
            temperature: self.cfg.temperature.unwrap_or(0.0),
            max_tokens: self.cfg.max_tokens.unwrap_or(1024),
            ..Default::default()
        }
    }

    fn to_item(&self, subject: &str, shots: &[McQuestion], q: McQuestion) -> Item {
        let prompt = build_prompt(subject, shots, &q);
        Item {
            id: q.id.clone(),
            group: q.group.clone(),
            messages: vec![Message::user(prompt)],
            params: self.params(),
            reference: json!({"answer": q.answer, "n": q.options.len()}),
        }
    }

    async fn load_mmlu(&self, data: &DataSource) -> Result<Vec<Item>> {
        let subset = self.cfg.subset.clone().unwrap_or_else(|| "all".into());
        let shots_n = self.cfg.shots.unwrap_or(5);
        let test = main_rows(&self.cfg, data, "cais/mmlu", &subset, "test").await?;
        let dev = if shots_n > 0 {
            data.rows("cais/mmlu", &subset, "dev").await?
        } else {
            Vec::new()
        };
        let parse = |r: &Value, i: usize| -> Result<McQuestion> {
            Ok(McQuestion {
                id: format!("mmlu-{i}"),
                group: str_field(r, "subject")?,
                question: str_field(r, "question")?,
                options: str_list(r, "choices")?,
                answer: r["answer"]
                    .as_u64()
                    .ok_or_else(|| Error::Data("mmlu row without answer".into()))?
                    as usize,
            })
        };
        let dev_q: Vec<McQuestion> = dev
            .iter()
            .enumerate()
            .map(|(i, r)| parse(r, i))
            .collect::<Result<_>>()?;
        let mut items = Vec::new();
        for (i, r) in test.iter().enumerate() {
            let q = parse(r, i)?;
            let subject = q.group.replace('_', " ");
            let shots: Vec<McQuestion> = dev_q
                .iter()
                .filter(|d| d.group == q.group)
                .take(shots_n)
                .cloned()
                .collect();
            items.push(self.to_item(&subject, &shots, q));
        }
        Ok(items)
    }

    async fn load_mmlu_pro(&self, data: &DataSource) -> Result<Vec<Item>> {
        let shots_n = self.cfg.shots.unwrap_or(5);
        let test = main_rows(&self.cfg, data, "TIGER-Lab/MMLU-Pro", "default", "test").await?;
        let val = if shots_n > 0 {
            data.rows("TIGER-Lab/MMLU-Pro", "default", "validation")
                .await?
        } else {
            Vec::new()
        };
        let parse = |r: &Value, prefix: &str, i: usize| -> Result<McQuestion> {
            Ok(McQuestion {
                id: format!("{prefix}-{i}"),
                group: str_field(r, "category")?,
                question: str_field(r, "question")?,
                options: str_list(r, "options")?,
                answer: r["answer_index"]
                    .as_u64()
                    .ok_or_else(|| Error::Data("mmlu-pro row without answer_index".into()))?
                    as usize,
            })
        };
        let val_q: Vec<McQuestion> = val
            .iter()
            .enumerate()
            .map(|(i, r)| parse(r, "val", i))
            .collect::<Result<_>>()?;
        let mut items = Vec::new();
        for (i, r) in test.iter().enumerate() {
            let q = parse(r, "mmlu-pro", i)?;
            let shots: Vec<McQuestion> = val_q
                .iter()
                .filter(|d| d.group == q.group)
                .take(shots_n)
                .cloned()
                .collect();
            let subject = q.group.clone();
            items.push(self.to_item(&subject, &shots, q));
        }
        Ok(items)
    }

    async fn load_arc(&self, data: &DataSource) -> Result<Vec<Item>> {
        let subset = self
            .cfg
            .subset
            .clone()
            .unwrap_or_else(|| "ARC-Challenge".into());
        let rows = main_rows(&self.cfg, data, "allenai/ai2_arc", &subset, "test").await?;
        let mut items = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            let labels = str_list(&r["choices"], "label")?;
            let key = str_field(r, "answerKey")?;
            let answer = labels
                .iter()
                .position(|l| *l == key)
                .ok_or_else(|| Error::Data(format!("arc row {i}: answerKey not in labels")))?;
            let q = McQuestion {
                id: format!("arc-{i}"),
                group: subset.clone(),
                question: str_field(r, "question")?,
                options: str_list(&r["choices"], "text")?,
                answer,
            };
            items.push(self.to_item("science", &[], q));
        }
        Ok(items)
    }

    async fn load_gpqa(&self, data: &DataSource) -> Result<Vec<Item>> {
        let subset = self
            .cfg
            .subset
            .clone()
            .unwrap_or_else(|| "gpqa_diamond".into());
        let rows = main_rows(&self.cfg, data, "Idavidrein/gpqa", &subset, "train").await?;
        let mut items = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            let mut options = vec![
                str_field(r, "Correct Answer")?,
                str_field(r, "Incorrect Answer 1")?,
                str_field(r, "Incorrect Answer 2")?,
                str_field(r, "Incorrect Answer 3")?,
            ];
            let id = format!("gpqa-{i}");
            let answer = shuffle_with_answer(&mut options, 0, &id);
            let group = r["High-level domain"].as_str().unwrap_or("all").to_string();
            let q = McQuestion {
                id,
                group,
                question: str_field(r, "Question")?,
                options,
                answer,
            };
            items.push(self.to_item("graduate-level science", &[], q));
        }
        Ok(items)
    }

    async fn load_truthfulqa_mc1(&self, data: &DataSource) -> Result<Vec<Item>> {
        let rows = main_rows(
            &self.cfg,
            data,
            "truthfulqa/truthful_qa",
            "multiple_choice",
            "validation",
        )
        .await?;
        let mut items = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            let mut options = str_list(&r["mc1_targets"], "choices")?;
            let labels: Vec<u64> = r["mc1_targets"]["labels"]
                .as_array()
                .ok_or_else(|| Error::Data("truthfulqa row without labels".into()))?
                .iter()
                .filter_map(|v| v.as_u64())
                .collect();
            let correct = labels
                .iter()
                .position(|l| *l == 1)
                .ok_or_else(|| Error::Data(format!("truthfulqa row {i} has no correct label")))?;
            let id = format!("truthfulqa-mc1-{i}");
            // The dataset lists the correct answer first, so shuffle to
            // remove position bias.
            let answer = shuffle_with_answer(&mut options, correct, &id);
            let q = McQuestion {
                id,
                group: "all".into(),
                question: str_field(r, "question")?,
                options,
                answer,
            };
            items.push(self.to_item("common misconceptions and facts", &[], q));
        }
        Ok(items)
    }
}

/// Deterministically shuffle options, returning the new index of the answer.
pub fn shuffle_with_answer(options: &mut Vec<String>, answer: usize, id: &str) -> usize {
    let mut idx: Vec<usize> = (0..options.len()).collect();
    SplitMix64::new(seed_for(id, 0)).shuffle(&mut idx);
    let shuffled: Vec<String> = idx.iter().map(|i| options[*i].clone()).collect();
    let new_answer = idx.iter().position(|i| *i == answer).unwrap();
    *options = shuffled;
    new_answer
}

fn str_field(r: &Value, key: &str) -> Result<String> {
    r[key]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| Error::Data(format!("row missing string field '{key}'")))
}

fn str_list(r: &Value, key: &str) -> Result<Vec<String>> {
    r[key]
        .as_array()
        .ok_or_else(|| Error::Data(format!("row missing list field '{key}'")))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| Error::Data(format!("non-string entry in '{key}'")))
        })
        .collect()
}

#[async_trait]
impl Task for McTask {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> &str {
        &self.cfg.kind
    }
    fn primary_metric(&self) -> &str {
        "accuracy"
    }
    fn limit(&self) -> Option<usize> {
        self.cfg.limit
    }

    async fn load(&self, data: &DataSource) -> Result<Vec<Item>> {
        match self.cfg.kind.as_str() {
            "mmlu" => self.load_mmlu(data).await,
            "mmlu_pro" => self.load_mmlu_pro(data).await,
            "arc" => self.load_arc(data).await,
            "gpqa" => self.load_gpqa(data).await,
            "truthfulqa" => self.load_truthfulqa_mc1(data).await,
            k => Err(Error::Config(format!("not a multiple choice task: {k}"))),
        }
    }

    async fn score(&self, item: &Item, completions: &[Completion]) -> ItemScore {
        let n = item.reference["n"].as_u64().unwrap_or(4) as usize;
        let answer = item.reference["answer"].as_u64().unwrap_or(0) as usize;
        let predicted = completions.first().and_then(|c| extract_choice(&c.text, n));
        let correct = predicted == Some(answer);
        ItemScore::single(
            "accuracy",
            if correct { 1.0 } else { 0.0 },
            json!({
                "predicted": predicted.map(|i| (LETTERS[i] as char).to_string()),
                "answer": (LETTERS[answer] as char).to_string(),
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_stated_answers() {
        assert_eq!(extract_choice("Reasoning...\nAnswer: C", 4), Some(2));
        assert_eq!(extract_choice("The answer is (B).", 4), Some(1));
        assert_eq!(extract_choice("**Answer: D**", 4), Some(3));
        assert_eq!(extract_choice("Answer: A\nWait. Answer: B", 4), Some(1));
        assert_eq!(extract_choice("answer is b", 4), None);
    }

    #[test]
    fn extracts_leading_letter_only_when_bare() {
        assert_eq!(extract_choice("B", 4), Some(1));
        assert_eq!(extract_choice("(C) because", 4), Some(2));
        assert_eq!(extract_choice("A. the first one", 4), Some(0));
        // an article at the start of a sentence is not an answer
        assert_eq!(extract_choice("A cat is an animal", 4), None);
    }

    #[test]
    fn rejects_out_of_range_letters() {
        assert_eq!(extract_choice("Answer: F", 4), None);
        assert_eq!(extract_choice("Answer: F", 6), Some(5));
        assert_eq!(extract_choice("", 4), None);
    }

    #[test]
    fn prompt_layout() {
        let q = McQuestion {
            id: "1".into(),
            group: "g".into(),
            question: "2+2?".into(),
            options: vec!["3".into(), "4".into()],
            answer: 1,
        };
        let shot = McQuestion {
            question: "1+1?".into(),
            options: vec!["2".into(), "5".into()],
            answer: 0,
            ..q.clone()
        };
        let p = build_prompt("math", &[shot], &q);
        assert!(p.contains("about math."));
        assert!(p.contains("1+1?\nA. 2\nB. 5\nAnswer: A\n\n2+2?\nA. 3\nB. 4\nAnswer:"));
    }

    #[test]
    fn shuffle_keeps_answer_tracking() {
        let mut opts: Vec<String> = ["right", "x", "y", "z"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let a = shuffle_with_answer(&mut opts, 0, "id-1");
        assert_eq!(opts[a], "right");
        let mut again: Vec<String> = ["right", "x", "y", "z"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(shuffle_with_answer(&mut again, 0, "id-1"), a);
    }
}
