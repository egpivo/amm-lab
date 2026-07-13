//! Causal estimation layer (phase 1): the primary matched-overlap event-study
//! difference-in-differences for the frozen `Panel`, with two-way fixed effects,
//! token-pair cluster-robust inference, and a wild cluster bootstrap.
//!
//! Scope is deliberately ordered. Implemented: nearest-neighbour caliper matching
//! ([`matching`]), the event-time design ([`design`]), two-way FE residualization
//! ([`fe`]), OLS with cluster-robust variance and the wild cluster bootstrap ([`ols`]),
//! and the orchestration + estimator manifest ([`event_study`]). Deferred to later
//! phases, added only with explicit numerical checks: Honest-DiD-style parallel-trend
//! bounds, the omitted-confounding robustness value, and entropy balancing. The layer is
//! self-contained Rust (no R/Python runtime) so the estimate is reproducible and
//! deployable alongside the evidence layer in [`crate::data`].

pub mod adapter;
pub mod design;
pub mod design_meta;
pub mod event_study;
pub mod fe;
pub mod matching;
pub mod ols;

pub use adapter::{
    TreatmentMeta, WeekGrid, build_primary_event_study_data, check_roles_against_matches,
    panel_to_obs, units_from_meta,
};
pub use design_meta::{load_matched_pairs, load_treatment_meta};
pub use event_study::{
    EventStudyData, EventStudyResult, SampleComposition, build_matched_sample, run,
};
pub use matching::{MatchResult, Unit, nn_caliper_match, smd};
pub use ols::CoefInference;
