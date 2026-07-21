//! Deterministic PSTT library for future replication tooling.
//!
//! Frozen Python generators under `.local/selection_paper/` remain the paper
//! evidence authority. Modules here are additive parity implementations and
//! must not be cited as the historical generator of frozen results.
//!
//! P0 covers schemas, timestamp normalization, strict joins, orientation,
//! marks, weekly aggregation, diagnostics, panel assembly, and manifests.
//! Projection / bootstrap / acquisition belong to later gated phases.

pub mod application;
pub mod asof;
pub mod block_time;
pub mod bootstrap;
pub mod cex;
pub mod classification;
pub mod diagnostics;
pub mod error;
pub mod manifest;
pub mod marks;
pub mod orientation;
pub mod panel;
pub mod parity;
pub mod projection;
pub mod schema;
pub mod sensitivity;
pub mod weekly;

pub use error::{PsttError, Result};
