//! HumanEval code generation with sandboxed execution of generated code.
//!
//! Generated programs run in a separate Python process with an import
//! allowlist, a banned-call check, CPU and memory limits where the platform
//! supports `ulimit`, a wall-clock timeout and an empty temporary directory.

use std::collections::BTreeSet;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use openevals_core::config::TaskConfig;
use openevals_core::metrics::pass_at_k;
use openevals_core::{Completion, Error, GenParams, Message, Result};
use regex::Regex;
use serde_json::json;

use crate::data::DataSource;
use crate::task::{main_rows, Item, ItemScore, Task};

const ALLOWED_IMPORTS: &[&str] = &[
    "math",
    "random",
    "string",
    "itertools",
    "functools",
    "operator",
    "collections",
    "heapq",
    "bisect",
    "array",
    "copy",
    "pprint",
    "reprlib",
    "enum",
    "datetime",
    "calendar",
    "time",
    "json",
    "re",
    "difflib",
    "textwrap",
    "unicodedata",
    "stringprep",
    "struct",
    "codecs",
    "types",
    "weakref",
    "abc",
    "numbers",
    "cmath",
    "decimal",
    "fractions",
    "statistics",
    // Added relative to the Python implementation: HumanEval prompts import
    // typing, and one problem needs hashlib.
    "typing",
    "hashlib",
    "dataclasses",
    "__future__",
];

const BANNED_CALLS: &[&str] = &[
    r"__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"\bopen\s*\(",
    r"\bfile\s*\(",
    r"\binput\s*\(",
    r"\braw_input\s*\(",
    r"\bgetattr\s*\(",
    r"\bsetattr\s*\(",
    r"\bdelattr\s*\(",
    r"\bglobals\s*\(",
    r"\blocals\s*\(",
];

/// Static safety check. Returns the list of violations (empty when safe).
pub fn check_code_security(code: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let import_re =
        Regex::new(r"(?m)^\s*(?:import\s+(\w+)|from\s+(\w+)(?:\.\w+)*\s+import\b)").unwrap();
    let allowed: BTreeSet<&str> = ALLOWED_IMPORTS.iter().copied().collect();
    for c in import_re.captures_iter(code) {
        let m = c
            .get(1)
            .or_else(|| c.get(2))
            .map(|m| m.as_str())
            .unwrap_or("");
        if !allowed.contains(m) {
            violations.push(format!("Disallowed import: {m}"));
        }
    }
    for pat in BANNED_CALLS {
        if Regex::new(pat).unwrap().is_match(code) {
            violations.push(format!("Dangerous function call: {pat}"));
        }
    }
    violations
}

/// Build the program to test from the prompt and a model reply.
pub fn extract_code(prompt: &str, entry_point: &str, response: &str) -> String {
    let fence = Regex::new(r"(?s)```(?:python|py)?[ \t]*\n(.*?)```").unwrap();
    let def_marker = format!("def {entry_point}");
    if let Some(c) = fence.captures(response) {
        let block = &c[1];
        return if block.contains(&def_marker) {
            // Keep the prompt so helper functions and imports it defines stay
            // available; the block redefines the entry point afterwards.
            format!("{prompt}\n    pass\n\n{block}")
        } else {
            format!("{prompt}{block}")
        };
    }
    if response.contains(&def_marker) {
        return format!("{prompt}\n    pass\n\n{response}");
    }
    let mut cut = response.len();
    for w in ["\ndef", "\nclass", "\nif __name__", "\nprint"] {
        if let Some(i) = response.find(w) {
            cut = cut.min(i);
        }
    }
    format!("{prompt}{}", &response[..cut])
}

pub struct RunOutcome {
    pub passed: bool,
    pub error: Option<String>,
}

/// Run a test program under limits.
pub async fn execute(python: &str, program: &str, timeout_secs: u64) -> RunOutcome {
    let violations = check_code_security(program);
    if !violations.is_empty() {
        return RunOutcome {
            passed: false,
            error: Some(format!("Security violation: {}", violations.join("; "))),
        };
    }
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => {
            return RunOutcome {
                passed: false,
                error: Some(e.to_string()),
            }
        }
    };
    let file = dir.path().join("main.py");
    if let Err(e) = std::fs::write(&file, program) {
        return RunOutcome {
            passed: false,
            error: Some(e.to_string()),
        };
    }
    let script = format!(
        "ulimit -t {timeout_secs} 2>/dev/null; ulimit -v 1048576 2>/dev/null; exec \"$0\" -I main.py"
    );
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(script)
        .arg(python)
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn();
    let child = match child {
        Ok(c) => c,
        Err(e) => {
            return RunOutcome {
                passed: false,
                error: Some(format!("cannot start interpreter: {e}")),
            }
        }
    };
    match tokio::time::timeout(
        Duration::from_secs(timeout_secs + 2),
        child.wait_with_output(),
    )
    .await
    {
        Ok(Ok(out)) if out.status.success() => RunOutcome {
            passed: true,
            error: None,
        },
        Ok(Ok(out)) => RunOutcome {
            passed: false,
            error: Some(
                String::from_utf8_lossy(&out.stderr)
                    .chars()
                    .rev()
                    .take(500)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect(),
            ),
        },
        Ok(Err(e)) => RunOutcome {
            passed: false,
            error: Some(e.to_string()),
        },
        Err(_) => RunOutcome {
            passed: false,
            error: Some("Execution timed out".into()),
        },
    }
}

pub struct HumanEvalTask {
    cfg: TaskConfig,
    name: String,
}

impl HumanEvalTask {
    pub fn new(cfg: &TaskConfig) -> Self {
        Self {
            name: cfg.display_name(),
            cfg: cfg.clone(),
        }
    }
    fn n(&self) -> usize {
        self.cfg.samples.unwrap_or(1)
    }
    fn ks(&self) -> Vec<usize> {
        let mut ks = self.cfg.k_values.clone().unwrap_or_else(|| vec![1]);
        ks.retain(|k| *k >= 1 && *k <= self.n());
        if ks.is_empty() {
            ks.push(1);
        }
        ks
    }
}

#[async_trait]
impl Task for HumanEvalTask {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> &str {
        "humaneval"
    }
    fn primary_metric(&self) -> &str {
        "pass_at_1"
    }
    fn samples_per_item(&self) -> usize {
        self.n()
    }
    fn limit(&self) -> Option<usize> {
        self.cfg.limit
    }

    async fn load(&self, data: &DataSource) -> Result<Vec<Item>> {
        let rows = main_rows(
            &self.cfg,
            data,
            "openai/openai_humaneval",
            "openai_humaneval",
            "test",
        )
        .await?;
        // Single greedy sample by default; sampling is needed for pass@k, k > 1.
        let default_temp = if self.n() > 1 { 0.2 } else { 0.0 };
        rows.iter()
            .enumerate()
            .map(|(i, r)| {
                let get = |k: &str| {
                    r[k].as_str()
                        .map(String::from)
                        .ok_or_else(|| Error::Data(format!("humaneval row {i} missing '{k}'")))
                };
                let prompt = get("prompt")?;
                Ok(Item {
                    id: get("task_id").unwrap_or_else(|_| format!("humaneval-{i}")),
                    group: "all".into(),
                    messages: vec![Message::user(format!(
                        "Complete the following Python function. Reply with the complete function implementation in a single ```python code block.\n\n```python\n{prompt}\n```"
                    ))],
                    params: GenParams {
                        temperature: self.cfg.temperature.unwrap_or(default_temp),
                        max_tokens: self.cfg.max_tokens.unwrap_or(1024),
                        ..Default::default()
                    },
                    reference: json!({
                        "prompt": prompt,
                        "test": get("test")?,
                        "entry_point": get("entry_point")?,
                    }),
                })
            })
            .collect()
    }

    async fn score(&self, item: &Item, completions: &[Completion]) -> ItemScore {
        let prompt = item.reference["prompt"].as_str().unwrap_or("");
        let test = item.reference["test"].as_str().unwrap_or("");
        let entry = item.reference["entry_point"].as_str().unwrap_or("");
        let python = self.cfg.python.clone().unwrap_or_else(|| "python3".into());
        let timeout = self.cfg.timeout.unwrap_or(10);
        let mut passed = 0usize;
        let mut first_error = None;
        for c in completions {
            let program = format!(
                "{}\n{}\n\ncheck({})\n",
                extract_code(prompt, entry, &c.text),
                test,
                entry
            );
            let out = execute(&python, &program, timeout).await;
            if out.passed {
                passed += 1;
            } else if first_error.is_none() {
                first_error = out.error;
            }
        }
        let n = completions.len().max(1);
        let mut metrics = std::collections::BTreeMap::new();
        for k in self.ks() {
            metrics.insert(format!("pass_at_{k}"), pass_at_k(n, passed, k.min(n)));
        }
        ItemScore {
            metrics,
            detail: json!({"passed": passed, "n": completions.len(), "error": first_error}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROMPT: &str =
        "from typing import List\n\n\ndef add(a: int, b: int) -> int:\n    \"\"\"Add.\"\"\"\n";

    #[test]
    fn security_check() {
        assert!(check_code_security("import math\nfrom typing import List\nx = 1").is_empty());
        assert!(!check_code_security("import os\n").is_empty());
        assert!(!check_code_security("x = open('f')\n").is_empty());
        assert!(!check_code_security("eval('1')").is_empty());
        assert!(!check_code_security("from subprocess import run").is_empty());
        // prose in docstrings that starts with "from" or "import" is not an import
        assert!(
            check_code_security("    \"\"\"\n    from two integers, pick one\n    \"\"\"")
                .is_empty()
        );
        // identifiers that merely contain a banned word are fine
        assert!(check_code_security("def reopen_x(): return evaluate(1)").is_empty());
    }

    #[test]
    fn extracts_full_function_from_fence() {
        let reply = "Here:\n```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```\n";
        let code = extract_code(PROMPT, "add", reply);
        assert!(code.contains("return a + b"));
        assert!(code.starts_with("from typing import List"));
    }

    #[test]
    fn extracts_body_only_from_fence() {
        let reply = "```python\n    return a + b\n```";
        let code = extract_code(PROMPT, "add", reply);
        assert!(code.ends_with("    return a + b\n"));
    }

    #[test]
    fn unfenced_reply_is_truncated_at_stop_words() {
        let reply = "    return a + b\nprint(add(1, 2))";
        let code = extract_code(PROMPT, "add", reply);
        assert!(code.ends_with("    return a + b"));
        assert!(!code.contains("print"));
    }

    fn python_available() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn executes_passing_failing_and_looping_programs() {
        if !python_available() {
            return;
        }
        let ok = execute("python3", "assert 1 + 1 == 2\n", 5).await;
        assert!(ok.passed, "{:?}", ok.error);
        let bad = execute("python3", "assert 1 + 1 == 3\n", 5).await;
        assert!(!bad.passed);
        let slow = execute("python3", "while True:\n    pass\n", 1).await;
        assert!(!slow.passed);
        let sec = execute("python3", "import os\n", 5).await;
        assert!(sec.error.unwrap().contains("Security violation"));
    }

    #[tokio::test]
    async fn scores_samples_with_pass_at_k() {
        if !python_available() {
            return;
        }
        let task = HumanEvalTask::new(&TaskConfig {
            kind: "humaneval".into(),
            samples: Some(2),
            k_values: Some(vec![1, 2]),
            ..Default::default()
        });
        let item = Item {
            id: "HumanEval/0".into(),
            group: "all".into(),
            messages: vec![],
            params: GenParams::default(),
            reference: json!({
                "prompt": PROMPT,
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                "entry_point": "add",
            }),
        };
        let good = Completion {
            text: "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```".into(),
            ..Default::default()
        };
        let bad = Completion {
            text: "```python\ndef add(a: int, b: int) -> int:\n    return a - b\n```".into(),
            ..Default::default()
        };
        let s = task.score(&item, &[good, bad]).await;
        assert!((s.metrics["pass_at_1"] - 0.5).abs() < 1e-9);
        assert_eq!(s.metrics["pass_at_2"], 1.0);
    }
}
