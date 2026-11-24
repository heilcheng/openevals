//! Dataset access. Rows come from the Hugging Face datasets-server JSON API
//! and are cached on disk as JSONL, so later runs work offline. A JSONL file
//! placed in the data directory under the cache name is used as-is.

use std::path::{Path, PathBuf};

use openevals_core::{Error, Result};
use serde_json::Value;

const DEFAULT_SERVER: &str = "https://datasets-server.huggingface.co";
const PAGE: usize = 100;

pub struct DataSource {
    dir: PathBuf,
    token: Option<String>,
    server: String,
    http: reqwest::Client,
}

impl DataSource {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            token: std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()),
            server: DEFAULT_SERVER.into(),
            http: reqwest::Client::new(),
        }
    }

    pub fn with_server(mut self, server: &str) -> Self {
        self.server = server.trim_end_matches('/').to_string();
        self
    }

    pub fn cache_path(&self, dataset: &str, config: &str, split: &str) -> PathBuf {
        self.dir.join(format!(
            "{}.{}.{}.jsonl",
            dataset.replace('/', "__"),
            config,
            split
        ))
    }

    pub async fn rows(&self, dataset: &str, config: &str, split: &str) -> Result<Vec<Value>> {
        let path = self.cache_path(dataset, config, split);
        if path.exists() {
            return read_jsonl(&path);
        }
        let rows = self.fetch(dataset, config, split).await?;
        std::fs::create_dir_all(&self.dir)?;
        let tmp = path.with_extension("jsonl.tmp");
        let mut body = String::new();
        for r in &rows {
            body.push_str(&serde_json::to_string(r)?);
            body.push('\n');
        }
        std::fs::write(&tmp, body)?;
        std::fs::rename(&tmp, &path)?;
        Ok(rows)
    }

    async fn fetch(&self, dataset: &str, config: &str, split: &str) -> Result<Vec<Value>> {
        tracing::info!(dataset, config, split, "downloading dataset rows");
        let mut rows = Vec::new();
        let mut offset = 0usize;
        loop {
            let mut attempt = 0u32;
            let resp = loop {
                let mut rb = self.http.get(format!("{}/rows", self.server)).query(&[
                    ("dataset", dataset),
                    ("config", config),
                    ("split", split),
                    ("offset", &offset.to_string()),
                    ("length", &PAGE.to_string()),
                ]);
                if let Some(t) = &self.token {
                    rb = rb.bearer_auth(t);
                }
                let resp = rb.send().await.map_err(|e| Error::Data(e.to_string()))?;
                let status = resp.status();
                if status.is_success() {
                    break resp;
                }
                let transient = status.as_u16() == 429 || status.is_server_error();
                if transient && attempt < 8 {
                    let wait = resp
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(1u64 << attempt.min(5));
                    tracing::warn!(%status, wait, "dataset server busy, retrying");
                    tokio::time::sleep(std::time::Duration::from_secs(wait.min(60))).await;
                    attempt += 1;
                    continue;
                }
                let hint = if status.as_u16() == 401 || status.as_u16() == 403 {
                    " (gated datasets need HF_TOKEN with access granted)"
                } else {
                    ""
                };
                return Err(Error::Data(format!(
                    "{dataset}/{config}/{split}: server returned {status}{hint}"
                )));
            };
            let v: Value = resp.json().await.map_err(|e| Error::Data(e.to_string()))?;
            let page = v["rows"]
                .as_array()
                .ok_or_else(|| Error::Data("response has no rows".into()))?;
            for r in page {
                rows.push(r["row"].clone());
            }
            let total = v["num_rows_total"].as_u64().unwrap_or(0) as usize;
            offset += page.len();
            if page.is_empty() || offset >= total {
                break;
            }
        }
        Ok(rows)
    }
}

pub fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Data(format!("cannot read {}: {e}", path.display())))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str(l).map_err(|e| Error::Data(format!("{}: {e}", path.display())))
        })
        .collect()
}

/// Dataset locations used by the benchmarks, for the `download` command.
pub fn dataset_specs(kind: &str) -> Vec<(&'static str, &'static str, &'static str)> {
    match kind {
        "mmlu" => vec![("cais/mmlu", "all", "test"), ("cais/mmlu", "all", "dev")],
        "mmlu_pro" => vec![
            ("TIGER-Lab/MMLU-Pro", "default", "test"),
            ("TIGER-Lab/MMLU-Pro", "default", "validation"),
        ],
        "gsm8k" => vec![
            ("openai/gsm8k", "main", "test"),
            ("openai/gsm8k", "main", "train"),
        ],
        "math500" => vec![("HuggingFaceH4/MATH-500", "default", "test")],
        "humaneval" => vec![("openai/openai_humaneval", "openai_humaneval", "test")],
        "arc" => vec![("allenai/ai2_arc", "ARC-Challenge", "test")],
        "truthfulqa" => vec![
            ("truthfulqa/truthful_qa", "generation", "validation"),
            ("truthfulqa/truthful_qa", "multiple_choice", "validation"),
        ],
        "gpqa" => vec![("Idavidrein/gpqa", "gpqa_diamond", "train")],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reads_cached_file_without_network() {
        let dir = tempfile::tempdir().unwrap();
        let ds = DataSource::new(dir.path()).with_server("http://127.0.0.1:1");
        std::fs::write(
            ds.cache_path("a/b", "c", "test"),
            "{\"x\":1}\n\n{\"x\":2}\n",
        )
        .unwrap();
        let rows = ds.rows("a/b", "c", "test").await.unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["x"], 2);
    }

    #[tokio::test]
    async fn missing_dataset_with_unreachable_server_errors() {
        let dir = tempfile::tempdir().unwrap();
        let ds = DataSource::new(dir.path()).with_server("http://127.0.0.1:1");
        assert!(ds.rows("a/b", "c", "test").await.is_err());
    }
}
