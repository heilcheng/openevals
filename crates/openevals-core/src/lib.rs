//! Core types for openevals: generation requests, configuration, the model
//! registry, scoring metrics, caching and result formats.

pub mod cache;
pub mod config;
pub mod error;
pub mod metrics;
pub mod registry;
pub mod report;
pub mod results;
pub mod rng;
pub mod types;

pub use error::{Error, Result};
pub use types::{Completion, GenParams, GenRequest, Generator, Message};
