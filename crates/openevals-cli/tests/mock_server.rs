//! End to end test: the real binary against a mock OpenAI-compatible server
//! with dataset rows pre-seeded in the data directory (no network needed).

use std::net::SocketAddr;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

#[derive(Clone, Default)]
struct Mock {
    requests: Arc<AtomicUsize>,
    throttled: Arc<AtomicUsize>,
}

async fn chat(State(m): State<Mock>, Json(body): Json<Value>) -> Response {
    // Reject the very first request with 429 to exercise retries.
    if m.throttled.fetch_add(1, Ordering::SeqCst) == 0 {
        let mut h = HeaderMap::new();
        h.insert("retry-after", HeaderValue::from_static("0"));
        return (StatusCode::TOO_MANY_REQUESTS, h, "slow down").into_response();
    }
    m.requests.fetch_add(1, Ordering::SeqCst);
    assert_eq!(
        body["model"], "Qwen/Qwen3-8B",
        "registry hf_repo is the served name"
    );
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    let prompt = body["messages"][0]["content"].as_str().unwrap_or("");

    if body["stream"] == true {
        let chunks = [
            json!({"choices":[{"delta":{"content":"tok "}}]}),
            json!({"choices":[{"delta":{"content":"tok "}}]}),
            json!({"choices":[{"delta":{"content":"tok"},"finish_reason":"stop"}]}),
            json!({"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}),
        ];
        let mut sse = String::new();
        for c in chunks {
            sse.push_str(&format!("data: {c}\n\n"));
        }
        sse.push_str("data: [DONE]\n\n");
        return ([(header::CONTENT_TYPE, "text/event-stream")], sse).into_response();
    }

    let text = if prompt.contains("capital of France") {
        "Reasoning first.\nAnswer: B"
    } else if prompt.contains("Janet") {
        "She earns 16 * 2 = 32... actually\n#### 18"
    } else {
        "#### 0"
    };
    Json(json!({
        "choices":[{"message":{"content": text},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":10,"completion_tokens":4}
    }))
    .into_response()
}

async fn start() -> (SocketAddr, Mock) {
    let mock = Mock::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .with_state(mock.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (addr, mock)
}

fn write_rows(dir: &std::path::Path, name: &str, rows: &[Value]) {
    let body: String = rows.iter().map(|r| format!("{r}\n")).collect();
    std::fs::write(dir.join(name), body).unwrap();
}

async fn run_cli(args: Vec<String>) -> (bool, String, String) {
    tokio::task::spawn_blocking(move || {
        let out = Command::new(env!("CARGO_BIN_EXE_openevals"))
            .args(&args)
            .output()
            .unwrap();
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).to_string(),
            String::from_utf8_lossy(&out.stderr).to_string(),
        )
    })
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn full_run_resume_and_report() {
    let (addr, mock) = start().await;
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_rows(
        &data,
        "cais__mmlu.all.test.jsonl",
        &[
            json!({"question":"What is the capital of France?","subject":"geography","choices":["Rome","Paris","Berlin","Madrid"],"answer":1}),
            json!({"question":"What is the capital of Spain?","subject":"geography","choices":["Rome","Paris","Berlin","Madrid"],"answer":3}),
        ],
    );
    write_rows(
        &data,
        "openai__gsm8k.main.test.jsonl",
        &[
            json!({"question":"Janet's ducks lay 16 eggs. How many does she sell?","answer":"work\n#### 18"}),
            json!({"question":"What is 2+2?","answer":"#### 4"}),
        ],
    );

    let out = dir.path().join("out");
    let cfg = format!(
        r#"
[run]
output = "{out}"
data_dir = "{data}"
concurrency = 2
retries = 3

[endpoint]
base_url = "http://{addr}/v1"

[[models]]
model = "qwen3-8b"
thinking = false

[[tasks]]
type = "mmlu"
shots = 0

[[tasks]]
type = "gsm8k"
shots = 0

[[tasks]]
type = "efficiency"
output_lengths = [8]
sample_prompts = ["hello"]
warmup_runs = 0
measurement_runs = 2
"#,
        out = out.display(),
        data = data.display(),
        addr = addr
    );
    let cfg_path = dir.path().join("run.toml");
    std::fs::write(&cfg_path, cfg).unwrap();
    let args = |extra: &[&str]| -> Vec<String> {
        let mut v = vec![
            "run".to_string(),
            "-c".into(),
            cfg_path.display().to_string(),
        ];
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    };

    let (ok, stdout, stderr) = run_cli(args(&[])).await;
    assert!(ok, "stdout: {stdout}\nstderr: {stderr}");

    let results: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("results.json")).unwrap()).unwrap();
    let m = &results["models"]["qwen3-8b"];
    // Paris question answered "B" (correct), Spain question answered "B" (wrong)
    assert_eq!(m["mmlu"]["metrics"]["accuracy"], 0.5);
    assert_eq!(m["mmlu"]["n_items"], 2);
    // Janet answered 18 (correct), other answered 0 (wrong)
    assert_eq!(m["gsm8k"]["metrics"]["accuracy"], 0.5);
    let tps = m["efficiency"]["metrics"]["tokens_per_second"]
        .as_f64()
        .unwrap();
    assert!(tps > 0.0);
    assert!(m["efficiency"]["metrics"]["tokens_8.ttft_ms"].is_number());
    assert!(out.join("leaderboard.md").exists());
    assert!(out.join("samples/qwen3-8b/mmlu.jsonl").exists());

    // Resume: a second run must serve all scored requests from the cache.
    let before = mock.requests.load(Ordering::SeqCst);
    let (ok, _, stderr) = run_cli(args(&["--tasks", "mmlu", "gsm8k"])).await;
    assert!(ok, "{stderr}");
    assert_eq!(
        mock.requests.load(Ordering::SeqCst),
        before,
        "cache must satisfy reruns"
    );

    // Report rendering
    let (ok, md, _) = run_cli(vec!["report".into(), out.display().to_string()]).await;
    assert!(ok);
    assert!(md.contains("| 1 | qwen3-8b |"));
    let (ok, csv, _) = run_cli(vec![
        "report".into(),
        out.display().to_string(),
        "--format".into(),
        "csv".into(),
    ])
    .await;
    assert!(ok && csv.starts_with("rank,model,average"));
}

#[tokio::test]
async fn cli_lists_models_and_validates_config() {
    let (ok, out, _) = run_cli(vec!["models".into(), "--family".into(), "qwen".into()]).await;
    assert!(ok);
    assert!(out.contains("Qwen/Qwen3-8B"));
    let (ok, out, _) = run_cli(vec!["models".into(), "--format".into(), "json".into()]).await;
    assert!(ok);
    let v: Value = serde_json::from_str(&out).unwrap();
    assert!(v.as_array().unwrap().len() >= 50);

    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.toml");
    std::fs::write(&bad, "[[tasks]]\ntype = \"nope\"\n").unwrap();
    let (ok, _, err) = run_cli(vec![
        "validate".into(),
        "-c".into(),
        bad.display().to_string(),
    ])
    .await;
    assert!(!ok);
    assert!(err.contains("unknown task type"));
}
