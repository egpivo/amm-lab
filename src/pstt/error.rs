//! Typed errors for deterministic PSTT parsing, joins, and schema checks.

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PsttError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("schema error: {0}")]
    Schema(String),
    #[error("invariant violated: {0}")]
    Invariant(String),
    #[error("duplicate key: {0}")]
    DuplicateKey(String),
    #[error("missing join: {0}")]
    MissingJoin(String),
    #[error("output directory is not empty: {0}")]
    NonEmptyOutput(PathBuf),
    #[error("path does not exist: {0}")]
    MissingPath(PathBuf),
}

impl PsttError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }

    pub fn schema(msg: impl Into<String>) -> Self {
        Self::Schema(msg.into())
    }

    pub fn invariant(msg: impl Into<String>) -> Self {
        Self::Invariant(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, PsttError>;
