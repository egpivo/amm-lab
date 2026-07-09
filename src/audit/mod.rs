//! Channel audit for AMM fee evaluation (Paper C).
//!
//! The formal core of the method: the admissibility routing of Algorithm 1
//! (channel + observability + support -> claim label -> routing destination) and the
//! exact LP-welfare channel decomposition it attributes design-based effects through.
//! The empirical pipeline that instantiates this on the Uniswap protocol-fee switch
//! (event pull, matching, event-study estimation) lives in `scripts/paper_c/`.

pub mod admissibility;
pub mod channel;
pub mod decomposition;
