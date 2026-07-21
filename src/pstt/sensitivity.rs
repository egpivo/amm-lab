//! Hidden-mass sensitivity maps and the signed five-corner projection,
//! parity-matched to frozen `mc_m5s.py` (`d1r_signed_ranges`, `signed_true`)
//! and the standalone envelope rule in `build_m6_public.py` stage 2.

use crate::pstt::error::{PsttError, Result};
use crate::pstt::projection::{LinearFractional, lfrac_ext};
use nalgebra::{DMatrix, DVector};

/// Weekly mark/service envelope `(ell_lo, ell_hi, s_lo, s_hi)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Envelope {
    pub ell_lo: f64,
    pub ell_hi: f64,
    pub s_lo: f64,
    pub s_hi: f64,
}

impl Envelope {
    pub fn validate(self) -> Result<Self> {
        if ![self.ell_lo, self.ell_hi, self.s_lo, self.s_hi]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(PsttError::invariant("non-finite envelope"));
        }
        if self.ell_lo > self.ell_hi || self.s_lo > self.s_hi || self.s_lo < 0.0 {
            return Err(PsttError::invariant("reversed or negative envelope"));
        }
        Ok(self)
    }
}

/// Standalone M6 envelope rule (frozen stage 2):
/// `wm = L/max(Om,1)` per week, `ws = S/max(Om,1)`;
/// `env = (min wm, max wm, max(1e-9, min ws), max ws)`.
pub fn envelope_from_weekly(l: &[f64], s: &[f64], om: &[f64]) -> Result<Envelope> {
    if l.is_empty() || l.len() != s.len() || l.len() != om.len() {
        return Err(PsttError::invariant("empty or mismatched weekly arrays"));
    }
    let mut wm_min = f64::INFINITY;
    let mut wm_max = f64::NEG_INFINITY;
    let mut ws_min = f64::INFINITY;
    let mut ws_max = f64::NEG_INFINITY;
    for i in 0..l.len() {
        let denom = if om[i] > 0.0 { om[i] } else { 1.0 };
        let wm = l[i] / denom;
        let ws = s[i] / denom;
        wm_min = wm_min.min(wm);
        wm_max = wm_max.max(wm);
        ws_min = ws_min.min(ws);
        ws_max = ws_max.max(ws);
    }
    Envelope {
        ell_lo: wm_min,
        ell_hi: wm_max,
        s_lo: ws_min.max(1e-9),
        s_hi: ws_max,
    }
    .validate()
}

/// One sensitivity corner `(r, ell, s)` of the signed Prop-2 map
/// `M(theta) = (theta_L + r*ell*omega) / (theta_S + r*s*omega)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corner {
    pub r: f64,
    pub ell: f64,
    pub s: f64,
}

/// The frozen corner enumeration: the observed `r=0` corner exactly once,
/// plus the four `(r_bar, ell, s)` envelope corners when `r_bar > 0`.
/// At `r_bar = 0`, only the observed corner is emitted.
pub fn signed_corners(r_bar: f64, env: Envelope) -> Vec<Corner> {
    let mut corners = vec![Corner {
        r: 0.0,
        ell: 0.0,
        s: 0.0,
    }];
    if r_bar > 0.0 {
        for e in [env.ell_lo, env.ell_hi] {
            for s in [env.s_lo, env.s_hi] {
                corners.push(Corner {
                    r: r_bar,
                    ell: e,
                    s,
                });
            }
        }
    }
    corners
}

/// Signed projection outcome. Abstention reasons are typed, never erased.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignedProjection {
    Region {
        /// `M_lo` composite: (min of corner minima, min of corner maxima).
        m_lo: (f64, f64),
        /// `M_hi` composite: (max of corner minima, max of corner maxima).
        m_hi: (f64, f64),
    },
    /// Region-wise denominator screen failed: `inf_E theta_S/omega <= 0`.
    PositivityLimited,
    /// A corner's Fieller denominator confidence set reached zero.
    CornerUnbounded,
    NonFiniteInput,
}

impl SignedProjection {
    pub fn region(self) -> Option<((f64, f64), (f64, f64))> {
        match self {
            SignedProjection::Region { m_lo, m_hi } => Some((m_lo, m_hi)),
            _ => None,
        }
    }

    pub fn is_usable(self) -> bool {
        matches!(self, SignedProjection::Region { .. })
    }
}

/// Frozen `mc_m5s.d1r_signed_ranges` over `theta = (theta_L, theta_S, omega)`:
///
/// 1. Region-wise denominator screen: project `theta_S/omega` over the
///    ellipsoid; abstain unless its lower endpoint is strictly positive.
/// 2. Project every corner map; abstain if any corner is unbounded.
/// 3. Compose `M_lo = (min mins, min maxs)`, `M_hi = (max mins, max maxs)`.
pub fn d1r_signed_ranges(
    theta: &DVector<f64>,
    sigma: &DMatrix<f64>,
    c: f64,
    r_bar: f64,
    env: Envelope,
) -> SignedProjection {
    let screen = lfrac_ext(
        &[0.0, 1.0, 0.0],
        0.0,
        &[0.0, 0.0, 1.0],
        0.0,
        theta,
        sigma,
        c,
    );
    match screen {
        LinearFractional::Projected { lo, .. } if lo > 0.0 => {}
        LinearFractional::NonFiniteInput => return SignedProjection::NonFiniteInput,
        _ => return SignedProjection::PositivityLimited,
    }
    let mut c_mins = Vec::new();
    let mut c_maxs = Vec::new();
    for corner in signed_corners(r_bar, env) {
        let proj = lfrac_ext(
            &[1.0, 0.0, corner.r * corner.ell],
            0.0,
            &[0.0, 1.0, corner.r * corner.s],
            0.0,
            theta,
            sigma,
            c,
        );
        match proj {
            LinearFractional::Projected { lo, hi } => {
                c_mins.push(lo);
                c_maxs.push(hi);
            }
            LinearFractional::DenominatorTouchesZero => {
                return SignedProjection::CornerUnbounded;
            }
            LinearFractional::NonFiniteInput => return SignedProjection::NonFiniteInput,
        }
    }
    let fold_min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let fold_max = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    SignedProjection::Region {
        m_lo: (fold_min(&c_mins), fold_min(&c_maxs)),
        m_hi: (fold_max(&c_mins), fold_max(&c_maxs)),
    }
}

/// Contrast interval composition (frozen standalone ranking and M5-R rank):
/// `I_Delta = [lower.Mlo.lo - higher.Mhi.hi, lower.Mhi.hi - higher.Mlo.lo]`.
pub fn contrast_interval(
    lower_m_lo: (f64, f64),
    lower_m_hi: (f64, f64),
    higher_m_lo: (f64, f64),
    higher_m_hi: (f64, f64),
) -> (f64, f64) {
    (lower_m_lo.0 - higher_m_hi.1, lower_m_hi.1 - higher_m_lo.0)
}

/// Frozen `mc_m5s.signed_true`: population identified-set endpoints.
pub fn signed_true(mu_l: f64, mu_s: f64, r_bar: f64, env: Envelope) -> (f64, f64) {
    let mut vals = vec![mu_l / mu_s];
    for e in [env.ell_lo, env.ell_hi] {
        for s in [env.s_lo, env.s_hi] {
            vals.push((mu_l + r_bar * e) / (mu_s + r_bar * s));
        }
    }
    let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (lo, hi)
}

/// `(M_lo, M_hi)` composite pair produced by [`d1r_signed_ranges`].
pub type RegionComposites = ((f64, f64), (f64, f64));

/// Verify monotone nesting: for `r1 <= r2`, `region(r1) ⊆ region(r2)`,
/// i.e. outer endpoints weakly widen along the grid.
pub fn monotone_nesting_holds(regions: &[(f64, RegionComposites)]) -> bool {
    let mut sorted: Vec<_> = regions.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for w in sorted.windows(2) {
        let (_, (lo1, hi1)) = w[0];
        let (_, (lo2, hi2)) = w[1];
        // Outer set endpoints: [Mlo.lo, Mhi.hi] must widen.
        if lo2.0 > lo1.0 + 1e-12 || hi2.1 < hi1.1 - 1e-12 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_counts_follow_frozen_rule() {
        let env = Envelope {
            ell_lo: -8.0,
            ell_hi: -4.0,
            s_lo: 10.0,
            s_hi: 30.0,
        };
        assert_eq!(signed_corners(0.0, env).len(), 1);
        assert_eq!(signed_corners(1.0, env).len(), 5);
    }

    #[test]
    fn signed_true_matches_rational_hand_values() {
        // S1 cell: muL=-50, muS=20, r_bar=1, ell in [-8,-4], s in [10,30].
        let env = Envelope {
            ell_lo: -8.0,
            ell_hi: -4.0,
            s_lo: 10.0,
            s_hi: 30.0,
        };
        let (lo, hi) = signed_true(-50.0, 20.0, 1.0, env);
        // corners: -50/20=-2.5, -58/30, -58/50, -54/30, -54/50
        assert!((lo - (-50.0 / 20.0)).abs() < 1e-12);
        assert!((hi - (-54.0 / 50.0)).abs() < 1e-12);
    }
}
