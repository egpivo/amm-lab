//! Algorithm 1: fee-claim admissibility routing (Paper C, Section 2.5).
//!
//! Deterministic routing over observability and support, not an estimator: it decides
//! *where* a claim can be answered.

use crate::audit::channel::{Channel, Observability, Support, observability};

/// The strongest admissible evidence a claim can carry, ordered by the
/// data--assumption tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    Measurement,
    DesignBasedCausal,
    ModelConditioned,
    NonEstimand,
}

/// Where a labelled claim is answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    DataReconstruction,
    EmpiricalDesign,
    DiagnosticModel,
    BoundaryStatement,
}

impl Label {
    pub fn route(self) -> Route {
        match self {
            Label::Measurement => Route::DataReconstruction,
            Label::DesignBasedCausal => Route::EmpiricalDesign,
            Label::ModelConditioned => Route::DiagnosticModel,
            Label::NonEstimand => Route::BoundaryStatement,
        }
    }
}

/// A fee-evaluation claim reduced to the inputs Algorithm 1 reads.
pub struct Claim {
    /// The intervention the claim requires.
    pub intervention: Channel,
    /// True if the claim reconstructs a quantity with no counterfactual content.
    pub non_counterfactual: bool,
    /// Support of the required intervention in the record.
    pub support: Support,
    /// True if diagnosable comparison units exist for that variation.
    pub comparison_units: bool,
}

/// Algorithm 1: assign the strongest admissible label and its routing destination.
pub fn audit(claim: &Claim) -> (Label, Route) {
    let label = if claim.non_counterfactual {
        Label::Measurement
    } else if claim.support != Support::None && claim.comparison_units {
        Label::DesignBasedCausal
    } else if observability(claim.intervention) == Observability::Latent {
        Label::ModelConditioned
    } else {
        Label::NonEstimand
    };
    (label, label.route())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::channel::support_under_fee_switch;

    #[test]
    fn take_rate_claim_is_design_based() {
        // "protocol fees reduce LP supply": rho varies at pool level, controls exist.
        let claim = Claim {
            intervention: Channel::TakeRate,
            non_counterfactual: false,
            support: support_under_fee_switch(Channel::TakeRate),
            comparison_units: true,
        };
        assert_eq!(
            audit(&claim),
            (Label::DesignBasedCausal, Route::EmpiricalDesign)
        );
    }

    #[test]
    fn trader_fee_claim_is_non_estimand() {
        // "a dynamic fee reduces adverse selection": c has no within-pool variation.
        let claim = Claim {
            intervention: Channel::TraderFee,
            non_counterfactual: false,
            support: support_under_fee_switch(Channel::TraderFee),
            comparison_units: false,
        };
        // TraderFee is not latent itself, so with no support it routes to a boundary.
        assert_eq!(
            audit(&claim),
            (Label::NonEstimand, Route::BoundaryStatement)
        );
    }

    #[test]
    fn latent_channel_without_support_is_model_conditioned() {
        // A claim on LVR attribution: latent, no design variation.
        let claim = Claim {
            intervention: Channel::Lvr,
            non_counterfactual: false,
            support: Support::None,
            comparison_units: false,
        };
        assert_eq!(
            audit(&claim),
            (Label::ModelConditioned, Route::DiagnosticModel)
        );
    }
}
