//! Dynamic fee rules for competing pools.
//!
//! The linearized rule is *equilibrium-inspired*, not a solved equilibrium:
//! it is a first-order approximation of the dynamic-fee structure in
//! arXiv:2603.09669 (duopoly competition) and arXiv:2506.02869 (single-pool
//! two-regime behavior). Whether it actually behaves in the expected
//! two-regime way is a diagnostic, not an assumption.

use serde::{Deserialize, Serialize};

/// Inputs to a pool's fee rule at the start of a step.
#[derive(Debug, Clone, Copy)]
pub struct FeeInputs {
    /// Own pool inventory imbalance in [-1, 1] (positive = X-heavy).
    pub own_imbalance: f64,
    /// Rival pool inventory imbalance (0.0 in monopoly mode).
    pub rival_imbalance: f64,
    /// (own_mid - oracle) / oracle.
    pub oracle_misalignment: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct FeePair {
    pub buy: f64,
    pub sell: f64,
}

pub trait FeeRule {
    fn name(&self) -> &'static str;
    fn fees(&self, inputs: &FeeInputs) -> FeePair;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantFee {
    pub fee: f64,
}

impl FeeRule for ConstantFee {
    fn name(&self) -> &'static str {
        "constant"
    }
    fn fees(&self, _inputs: &FeeInputs) -> FeePair {
        FeePair {
            buy: self.fee,
            sell: self.fee,
        }
    }
}

/// fee_buy  = base + a_own*own + a_rival*rival + a_oracle*misalignment
/// fee_sell = base - a_own*own - a_rival*rival - a_oracle*misalignment
/// both clipped to [min_fee, max_fee].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearDynamicFee {
    pub base_fee: f64,
    pub a_own: f64,
    pub a_rival: f64,
    pub a_oracle: f64,
    pub min_fee: f64,
    pub max_fee: f64,
}

impl LinearDynamicFee {
    pub fn duopoly_default() -> Self {
        Self {
            base_fee: 0.003,
            a_own: 0.01,
            a_rival: -0.005,
            a_oracle: 0.5,
            min_fee: 0.0001,
            max_fee: 0.05,
        }
    }
}

impl FeeRule for LinearDynamicFee {
    fn name(&self) -> &'static str {
        "linear_dynamic"
    }
    fn fees(&self, inputs: &FeeInputs) -> FeePair {
        let tilt = self.a_own * inputs.own_imbalance
            + self.a_rival * inputs.rival_imbalance
            + self.a_oracle * inputs.oracle_misalignment;
        FeePair {
            buy: (self.base_fee + tilt).clamp(self.min_fee, self.max_fee),
            sell: (self.base_fee - tilt).clamp(self.min_fee, self.max_fee),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misalignment_tilts_fees_asymmetrically() {
        let rule = LinearDynamicFee::duopoly_default();
        // Pool mid above oracle: buying Y from the pool is expensive for
        // arbs anyway; selling Y to the pool is the arb direction, so the
        // rule should tilt buy/sell fees apart.
        let inp = FeeInputs {
            own_imbalance: 0.0,
            rival_imbalance: 0.0,
            oracle_misalignment: 0.01,
        };
        let f = rule.fees(&inp);
        assert!(f.buy > f.sell);
    }

    #[test]
    fn fees_are_clipped() {
        let rule = LinearDynamicFee::duopoly_default();
        let inp = FeeInputs {
            own_imbalance: 1.0,
            rival_imbalance: -1.0,
            oracle_misalignment: 1.0,
        };
        let f = rule.fees(&inp);
        assert!(f.buy <= rule.max_fee && f.sell >= rule.min_fee);
    }
}
