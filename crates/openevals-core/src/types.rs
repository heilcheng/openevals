use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
}

/// Sampling and length parameters for one request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GenParams {
    pub temperature: f64,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f64>,
    pub max_tokens: u32,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
            top_k: None,
            min_p: None,
            max_tokens: 1024,
            stop: Vec::new(),
            seed: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GenRequest {
    pub messages: Vec<Message>,
    pub params: GenParams,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct Completion {
    /// Final answer text with any reasoning block removed.
    pub text: String,
    /// Reasoning content, when the server returns it separately or the text
    /// contains a think block.
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    #[serde(default)]
    pub latency_ms: f64,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    /// Set by the runner when generation failed after all retries.
    #[serde(default)]
    pub error: Option<String>,
}

#[async_trait]
pub trait Generator: Send + Sync {
    /// Identifier used in cache keys and result files.
    fn model_id(&self) -> &str;

    /// Everything about the backend that changes outputs but is not part of
    /// the request (served name, thinking switches, extra body fields). It is
    /// mixed into cache keys so that changing it never reuses stale replies.
    fn fingerprint(&self) -> String {
        self.model_id().to_string()
    }

    async fn generate(&self, req: &GenRequest) -> Result<Completion>;

    /// Streaming generation that records time to first token. Backends
    /// without streaming fall back to `generate`.
    async fn generate_timed(&self, req: &GenRequest) -> Result<Completion> {
        self.generate(req).await
    }
}

/// Split reasoning from the answer for models that inline `<think>` blocks.
/// Returns `(answer, reasoning)`.
pub fn split_think(text: &str) -> (String, Option<String>) {
    const CLOSE: &str = "</think>";
    if let Some(pos) = text.rfind(CLOSE) {
        let reasoning = text[..pos].replace("<think>", "").trim().to_string();
        let answer = text[pos + CLOSE.len()..].trim().to_string();
        return (answer, Some(reasoning));
    }
    if let Some(pos) = text.find("<think>") {
        // Unterminated block: generation was cut off inside the reasoning.
        let reasoning = text[pos + "<think>".len()..].trim().to_string();
        return (text[..pos].trim().to_string(), Some(reasoning));
    }
    (text.to_string(), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_closed_think_block() {
        let (a, r) = split_think("<think>step one</think>\n\nFinal: 4");
        assert_eq!(a, "Final: 4");
        assert_eq!(r.as_deref(), Some("step one"));
    }

    #[test]
    fn handles_missing_open_tag() {
        let (a, r) = split_think("reasoning here</think>42");
        assert_eq!(a, "42");
        assert_eq!(r.as_deref(), Some("reasoning here"));
    }

    #[test]
    fn unterminated_block_gives_empty_answer() {
        let (a, r) = split_think("<think>still thinking");
        assert_eq!(a, "");
        assert_eq!(r.as_deref(), Some("still thinking"));
    }

    #[test]
    fn plain_text_untouched() {
        assert_eq!(split_think("hello"), ("hello".to_string(), None));
    }
}
