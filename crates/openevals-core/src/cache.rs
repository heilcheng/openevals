//! Append-only JSONL cache of completions, keyed by a hash of the request.
//! This makes runs resumable and re-scoring free.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Completion, GenRequest, Result};

#[derive(Serialize, Deserialize)]
struct Line {
    key: String,
    completion: Completion,
}

pub struct Cache {
    map: Mutex<HashMap<String, Completion>>,
    file: Mutex<File>,
}

/// Stable key over model, request contents and run index.
pub fn request_key(model_key: &str, req: &GenRequest, run: u32) -> String {
    let payload = serde_json::json!({
        "model": model_key,
        "request": req,
        "run": run,
    });
    let mut hasher = Sha256::new();
    hasher.update(payload.to_string().as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

impl Cache {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut map = HashMap::new();
        if path.exists() {
            let reader = BufReader::new(File::open(path)?);
            for line in reader.lines() {
                let line = line?;
                // A truncated final line from an interrupted run is skipped.
                if let Ok(entry) = serde_json::from_str::<Line>(&line) {
                    map.insert(entry.key, entry.completion);
                }
            }
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            map: Mutex::new(map),
            file: Mutex::new(file),
        })
    }

    pub fn get(&self, key: &str) -> Option<Completion> {
        self.map.lock().unwrap().get(key).cloned()
    }

    pub fn put(&self, key: &str, completion: &Completion) -> Result<()> {
        let line = serde_json::to_string(&Line {
            key: key.to_string(),
            completion: completion.clone(),
        })?;
        {
            let mut f = self.file.lock().unwrap();
            writeln!(f, "{line}")?;
            f.flush()?;
        }
        self.map
            .lock()
            .unwrap()
            .insert(key.to_string(), completion.clone());
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.map.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenParams, Message};

    fn req(text: &str) -> GenRequest {
        GenRequest {
            messages: vec![Message::user(text)],
            params: GenParams::default(),
        }
    }

    #[test]
    fn key_changes_with_request_and_run() {
        let a = request_key("m", &req("a"), 0);
        assert_eq!(a, request_key("m", &req("a"), 0));
        assert_ne!(a, request_key("m", &req("b"), 0));
        assert_ne!(a, request_key("m", &req("a"), 1));
        assert_ne!(a, request_key("n", &req("a"), 0));
    }

    #[test]
    fn persists_and_reloads_ignoring_truncated_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.jsonl");
        {
            let c = Cache::open(&path).unwrap();
            c.put(
                "k1",
                &Completion {
                    text: "hi".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        }
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        write!(f, "{{\"key\":\"k2\",\"compl").unwrap();
        let c = Cache::open(&path).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c.get("k1").unwrap().text, "hi");
        assert!(c.get("k2").is_none());
    }
}
