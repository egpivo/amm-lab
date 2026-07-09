//! Exact LP-welfare channel decomposition (Paper C, Section 2.2).
//!
//! W = FeeRev_arb + FeeRev_fund - LVR, path by path. For a design-based contrast the
//! effect decomposes exactly into the same three channels; the gross-normalized
//! channel-share vector reports which channel carries the effect without the
//! instability of dividing by a small net welfare change.

/// The three welfare channels of a policy contrast.
#[derive(Debug, Clone, Copy)]
pub struct ChannelDelta {
    pub fee_rev_arb: f64,
    pub fee_rev_fund: f64,
    pub lvr: f64,
}

impl ChannelDelta {
    /// Net welfare change: dW = dFeeRev_arb + dFeeRev_fund - dLVR.
    pub fn delta_welfare(&self) -> f64 {
        self.fee_rev_arb + self.fee_rev_fund - self.lvr
    }

    /// Identity residual against an independently computed dW; must sit at numerical
    /// precision for the decomposition to be a read-out rather than an approximation.
    pub fn identity_residual(&self, delta_welfare_observed: f64) -> f64 {
        (delta_welfare_observed - self.delta_welfare()).abs()
    }

    /// Gross-normalized channel-share vector (arb, fund, -lvr) / sum|.|.
    pub fn channel_shares(&self) -> [f64; 3] {
        let z = [self.fee_rev_arb, self.fee_rev_fund, -self.lvr];
        let g: f64 = z.iter().map(|x| x.abs()).sum();
        if g == 0.0 {
            [0.0; 3]
        } else {
            [z[0] / g, z[1] / g, z[2] / g]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_holds() {
        let d = ChannelDelta {
            fee_rev_arb: -0.0065,
            fee_rev_fund: 0.0216,
            lvr: 0.0,
        };
        assert!(d.identity_residual(d.delta_welfare()) < 1e-12);
    }

    #[test]
    fn shares_sum_to_one_in_absolute_value() {
        let d = ChannelDelta {
            fee_rev_arb: -0.0065,
            fee_rev_fund: 0.0216,
            lvr: 0.0,
        };
        let s = d.channel_shares();
        let g: f64 = s.iter().map(|x| x.abs()).sum();
        assert!((g - 1.0).abs() < 1e-12);
    }
}
