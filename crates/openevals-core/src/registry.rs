//! The model registry (`models.toml`): researched open-weight models with
//! Hugging Face repo IDs, sizes, context lengths, licenses and recommended
//! sampling.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const EMBEDDED: &str = include_str!("../../../models.toml");

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Reasoning {
    #[default]
    None,
    Hybrid,
    Always,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct Sampling {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub min_p: Option<f64>,
    pub max_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub family: String,
    pub name: String,
    pub hf_repo: String,
    pub params_b: f64,
    pub active_params_b: f64,
    pub context_length: u64,
    pub license: String,
    #[serde(default)]
    pub gated: bool,
    #[serde(default)]
    pub reasoning: Reasoning,
    pub release: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub sampling_thinking: Option<Sampling>,
}

#[derive(Debug, Deserialize)]
struct File {
    #[serde(default)]
    model: Vec<ModelSpec>,
}

#[derive(Clone, Debug)]
pub struct Registry {
    models: Vec<ModelSpec>,
}

impl Registry {
    pub fn parse(text: &str) -> Result<Self> {
        let file: File = toml::from_str(text).map_err(|e| Error::Config(e.to_string()))?;
        let mut seen = BTreeMap::new();
        for m in &file.model {
            if seen.insert(m.id.clone(), ()).is_some() {
                return Err(Error::Config(format!("duplicate model id '{}'", m.id)));
            }
            if !m.hf_repo.contains('/') {
                return Err(Error::Config(format!(
                    "model '{}' has malformed hf_repo '{}'",
                    m.id, m.hf_repo
                )));
            }
        }
        Ok(Self { models: file.model })
    }

    /// Registry compiled into the binary.
    pub fn embedded() -> Self {
        Self::parse(EMBEDDED).expect("embedded models.toml must be valid")
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    pub fn all(&self) -> &[ModelSpec] {
        &self.models
    }

    pub fn get(&self, id: &str) -> Option<&ModelSpec> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Case-insensitive family filter (substring match).
    pub fn by_family(&self, family: &str) -> Vec<&ModelSpec> {
        let f = family.to_lowercase();
        self.models
            .iter()
            .filter(|m| m.family.to_lowercase().contains(&f))
            .collect()
    }
}

impl ModelSpec {
    /// Sampling to use given whether thinking is enabled.
    pub fn recommended_sampling(&self, thinking: bool) -> &Sampling {
        match (&self.sampling_thinking, thinking) {
            (Some(s), true) => s,
            _ => &self.sampling,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_registry_is_valid() {
        let r = Registry::embedded();
        assert!(r.all().len() >= 50, "got {}", r.all().len());
        assert!(r.get("qwen3-8b").is_some());
        for m in r.all() {
            assert!(m.params_b > 0.0, "{}", m.id);
            assert!(m.active_params_b <= m.params_b + 1e-9, "{}", m.id);
            assert!(m.context_length >= 2048, "{}", m.id);
            assert!(!m.license.is_empty(), "{}", m.id);
        }
    }

    #[test]
    fn rejects_duplicates_and_bad_repo() {
        let dup = "[[model]]\nid='a'\nfamily='f'\nname='n'\nhf_repo='o/r'\nparams_b=1\nactive_params_b=1\ncontext_length=4096\nlicense='mit'\nrelease='2026-01'\n";
        assert!(Registry::parse(dup).is_ok());
        assert!(Registry::parse(&format!("{dup}{dup}")).is_err());
        assert!(Registry::parse(&dup.replace("o/r", "norepo")).is_err());
    }

    #[test]
    fn family_filter() {
        let r = Registry::embedded();
        assert!(!r.by_family("qwen").is_empty());
        assert!(r.by_family("nonexistent-family").is_empty());
    }
}
