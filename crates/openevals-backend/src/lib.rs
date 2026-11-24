//! OpenAI-compatible chat completions backend.
//!
//! One client covers vLLM, SGLang, llama.cpp server, Ollama, TGI and hosted
//! APIs that expose `/v1/chat/completions`.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::StreamExt;
use openevals_core::types::split_think;
use openevals_core::{Completion, Error, GenRequest, Generator, Result};
use serde_json::{json, Map, Value};

#[derive(Clone)]
pub struct BackendConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    /// Model name sent to the server.
    pub served_name: String,
    /// Label used in caches and result files.
    pub label: String,
    pub timeout: Duration,
    pub max_retries: u32,
    /// First backoff delay; doubles per attempt up to `max_backoff`.
    pub base_backoff: Duration,
    pub max_backoff: Duration,
    pub thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub extra_body: Map<String, Value>,
}

// Manual Debug so the API key can never end up in a log line.
impl std::fmt::Debug for BackendConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("served_name", &self.served_name)
            .field("label", &self.label)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("thinking", &self.thinking)
            .field("reasoning_effort", &self.reasoning_effort)
            .finish_non_exhaustive()
    }
}

impl BackendConfig {
    pub fn new(base_url: &str, served_name: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: None,
            served_name: served_name.to_string(),
            label: served_name.to_string(),
            timeout: Duration::from_secs(600),
            max_retries: 5,
            base_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
            thinking: None,
            reasoning_effort: None,
            extra_body: Map::new(),
        }
    }
}

pub struct OpenAiBackend {
    cfg: BackendConfig,
    http: reqwest::Client,
}

impl OpenAiBackend {
    pub fn new(cfg: BackendConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::Http(e.to_string()))?;
        Ok(Self { cfg, http })
    }

    /// Build the JSON request body.
    pub fn build_body(&self, req: &GenRequest, stream: bool) -> Value {
        let p = &req.params;
        let mut body = Map::new();
        body.insert("model".into(), json!(self.cfg.served_name));
        body.insert("messages".into(), json!(req.messages));
        body.insert("temperature".into(), json!(p.temperature));
        body.insert("max_tokens".into(), json!(p.max_tokens));
        if let Some(v) = p.top_p {
            body.insert("top_p".into(), json!(v));
        }
        if let Some(v) = p.top_k {
            body.insert("top_k".into(), json!(v));
        }
        if let Some(v) = p.min_p {
            body.insert("min_p".into(), json!(v));
        }
        if !p.stop.is_empty() {
            body.insert("stop".into(), json!(p.stop));
        }
        if let Some(v) = p.seed {
            body.insert("seed".into(), json!(v));
        }
        if let Some(t) = self.cfg.thinking {
            body.insert(
                "chat_template_kwargs".into(),
                json!({ "enable_thinking": t }),
            );
        }
        if let Some(e) = &self.cfg.reasoning_effort {
            body.insert("reasoning_effort".into(), json!(e));
        }
        if stream {
            body.insert("stream".into(), json!(true));
            body.insert("stream_options".into(), json!({ "include_usage": true }));
        }
        for (k, v) in &self.cfg.extra_body {
            body.insert(k.clone(), v.clone());
        }
        Value::Object(body)
    }

    fn url(&self) -> String {
        format!("{}/chat/completions", self.cfg.base_url)
    }

    fn backoff(&self, attempt: u32, err: &Error) -> Duration {
        if let Error::Status {
            retry_after_secs: Some(s),
            ..
        } = err
        {
            return Duration::from_secs(*s).min(self.cfg.max_backoff);
        }
        let factor = 1u32 << attempt.min(16);
        (self.cfg.base_backoff * factor).min(self.cfg.max_backoff)
    }

    async fn send(&self, body: &Value) -> Result<reqwest::Response> {
        let mut rb = self.http.post(self.url()).json(body);
        if let Some(k) = &self.cfg.api_key {
            rb = rb.bearer_auth(k);
        }
        let resp = rb.send().await.map_err(|e| Error::Http(e.to_string()))?;
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let retry_after_secs = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        let text = resp.text().await.unwrap_or_default();
        Err(Error::Status {
            status: status.as_u16(),
            body: text.chars().take(500).collect(),
            retry_after_secs,
        })
    }

    async fn with_retries<F, Fut>(&self, mut op: F) -> Result<Completion>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<Completion>>,
    {
        let mut attempt = 0;
        loop {
            match op().await {
                Ok(c) => return Ok(c),
                Err(e) if e.is_retryable() && attempt < self.cfg.max_retries => {
                    let wait = self.backoff(attempt, &e);
                    tracing::warn!(attempt, ?wait, error = %e, "request failed, retrying");
                    tokio::time::sleep(wait).await;
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn once(&self, req: &GenRequest) -> Result<Completion> {
        let started = Instant::now();
        let resp = self.send(&self.build_body(req, false)).await?;
        let v: Value = resp.json().await.map_err(|e| Error::Http(e.to_string()))?;
        let mut c = parse_response(&v)?;
        c.latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(c)
    }

    async fn once_stream(&self, req: &GenRequest) -> Result<Completion> {
        let started = Instant::now();
        let resp = self.send(&self.build_body(req, true)).await?;
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut ttft: Option<f64> = None;
        let mut usage: Option<Value> = None;
        let mut finish: Option<String> = None;
        let mut chunks = 0u64;
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| Error::Http(e.to_string()))?;
            buf.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(pos) = buf.find('\n') {
                let line: String = buf.drain(..=pos).collect();
                let line = line.trim();
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" || data.is_empty() {
                    continue;
                }
                let v: Value = serde_json::from_str(data)
                    .map_err(|e| Error::Parse(format!("bad SSE chunk: {e}")))?;
                if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                    usage = Some(u.clone());
                }
                if let Some(choice) = v["choices"].get(0) {
                    let delta = &choice["delta"];
                    let piece = delta["content"].as_str().unwrap_or("");
                    let rpiece = delta["reasoning_content"]
                        .as_str()
                        .or_else(|| delta["reasoning"].as_str())
                        .unwrap_or("");
                    if (!piece.is_empty() || !rpiece.is_empty()) && ttft.is_none() {
                        ttft = Some(started.elapsed().as_secs_f64() * 1000.0);
                    }
                    if !piece.is_empty() || !rpiece.is_empty() {
                        chunks += 1;
                    }
                    content.push_str(piece);
                    reasoning.push_str(rpiece);
                    if let Some(f) = choice["finish_reason"].as_str() {
                        finish = Some(f.to_string());
                    }
                }
            }
        }
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        let (text, inline_reasoning) = split_think(&content);
        let reasoning = if reasoning.is_empty() {
            inline_reasoning
        } else {
            Some(reasoning)
        };
        let completion_tokens = usage
            .as_ref()
            .and_then(|u| u["completion_tokens"].as_u64())
            .or(Some(chunks));
        Ok(Completion {
            text,
            reasoning,
            finish_reason: finish,
            prompt_tokens: usage.as_ref().and_then(|u| u["prompt_tokens"].as_u64()),
            completion_tokens,
            latency_ms,
            ttft_ms: ttft,
            error: None,
        })
    }
}

/// Parse a non-streaming chat completion response.
pub fn parse_response(v: &Value) -> Result<Completion> {
    let choice = v["choices"]
        .get(0)
        .ok_or_else(|| Error::Parse("response has no choices".into()))?;
    let msg = &choice["message"];
    let raw = msg["content"].as_str().unwrap_or("");
    let (text, inline) = split_think(raw);
    let separate = msg["reasoning_content"]
        .as_str()
        .or_else(|| msg["reasoning"].as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    Ok(Completion {
        text,
        reasoning: separate.or(inline),
        finish_reason: choice["finish_reason"].as_str().map(|s| s.to_string()),
        prompt_tokens: v["usage"]["prompt_tokens"].as_u64(),
        completion_tokens: v["usage"]["completion_tokens"].as_u64(),
        latency_ms: 0.0,
        ttft_ms: None,
        error: None,
    })
}

#[async_trait]
impl Generator for OpenAiBackend {
    fn model_id(&self) -> &str {
        &self.cfg.label
    }

    fn fingerprint(&self) -> String {
        let extra = serde_json::to_string(&self.cfg.extra_body).unwrap_or_default();
        format!(
            "{}|thinking={:?}|effort={:?}|extra={}",
            self.cfg.served_name, self.cfg.thinking, self.cfg.reasoning_effort, extra
        )
    }

    async fn generate(&self, req: &GenRequest) -> Result<Completion> {
        self.with_retries(|| self.once(req)).await
    }

    async fn generate_timed(&self, req: &GenRequest) -> Result<Completion> {
        self.with_retries(|| self.once_stream(req)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openevals_core::{GenParams, Message};

    fn backend(f: impl FnOnce(&mut BackendConfig)) -> OpenAiBackend {
        let mut c = BackendConfig::new("http://x/v1/", "m");
        f(&mut c);
        OpenAiBackend::new(c).unwrap()
    }

    fn req() -> GenRequest {
        GenRequest {
            messages: vec![Message::user("hi")],
            params: GenParams {
                temperature: 0.6,
                top_p: Some(0.95),
                top_k: Some(20),
                max_tokens: 32,
                ..Default::default()
            },
        }
    }

    #[test]
    fn body_contains_sampling_and_template_kwargs() {
        let b = backend(|c| {
            c.thinking = Some(false);
            c.reasoning_effort = Some("low".into());
            c.extra_body
                .insert("repetition_penalty".into(), json!(1.05));
        });
        let body = b.build_body(&req(), false);
        assert_eq!(body["model"], "m");
        assert_eq!(body["top_k"], 20);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(body["reasoning_effort"], "low");
        assert_eq!(body["repetition_penalty"], 1.05);
        assert!(body.get("stream").is_none());
        let s = b.build_body(&req(), true);
        assert_eq!(s["stream"], true);
        assert_eq!(b.url(), "http://x/v1/chat/completions");
    }

    #[test]
    fn parses_reasoning_variants() {
        let v = json!({"choices":[{"message":{"content":"<think>a</think>B","reasoning_content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4}});
        let c = parse_response(&v).unwrap();
        assert_eq!(c.text, "B");
        assert_eq!(c.reasoning.as_deref(), Some("a"));
        assert_eq!(c.completion_tokens, Some(4));
        let v = json!({"choices":[{"message":{"content":"B","reasoning_content":"why"}}]});
        let c = parse_response(&v).unwrap();
        assert_eq!(c.reasoning.as_deref(), Some("why"));
        assert!(parse_response(&json!({"choices":[]})).is_err());
    }

    #[test]
    fn backoff_doubles_and_caps_and_honours_retry_after() {
        let b = backend(|c| {
            c.base_backoff = Duration::from_millis(100);
            c.max_backoff = Duration::from_millis(500);
        });
        let e = Error::Http("x".into());
        assert_eq!(b.backoff(0, &e), Duration::from_millis(100));
        assert_eq!(b.backoff(2, &e), Duration::from_millis(400));
        assert_eq!(b.backoff(5, &e), Duration::from_millis(500));
        let ra = Error::Status {
            status: 429,
            body: String::new(),
            retry_after_secs: Some(1),
        };
        assert_eq!(b.backoff(0, &ra), Duration::from_millis(500));
    }

    #[test]
    fn fingerprint_tracks_thinking_and_hides_key() {
        let a = backend(|c| c.thinking = Some(true));
        let b = backend(|c| c.thinking = Some(false));
        assert_ne!(a.fingerprint(), b.fingerprint());
        let k = backend(|c| c.api_key = Some("sk-secret".into()));
        assert!(!format!("{:?}", k.cfg).contains("sk-secret"));
        assert!(!k.fingerprint().contains("sk-secret"));
    }
}
