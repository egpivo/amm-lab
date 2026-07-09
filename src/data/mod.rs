//! Offline data-handling layer (batch, cached, versioned).
//!
//! This is the first layer of the pull -> reconstruct -> method -> dashboard pipeline:
//! raw on-chain logs are normalized and reconstructed into a *frozen* [`Panel`] of
//! pool-week outcomes, which the method layer (`crate::audit` and the design/estimation
//! code) consumes without ever touching the network. The layer boundary is the typed
//! [`Panel`] schema in [`panel`]; the method and dashboard layers depend only on it.
//!
//! Status: schema, tick-book, and parity comparator are implemented and unit-tested.
//! The event -> panel reconstruction is being ported from the reference Python
//! (`build_outcomes.py` v3) and validated row-for-row against its frozen panel as a
//! golden reference (see [`panel::compare`]).

pub mod book;
pub mod completeness;
pub mod io;
pub mod panel;
pub mod reconstruct;

pub use book::Book;
pub use completeness::{CompletenessReport, frozen_week_grid, report as completeness_report};
pub use io::{
    ReconstructInputs, load_inputs, read_events, reconstruct_dir_streaming, reconstruct_streaming,
};
pub use panel::{Panel, ParityReport, PoolWeek, UnitRole, compare};
pub use reconstruct::{Event, reconstruct, reconstruct_pool};
