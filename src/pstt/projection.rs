//! Fieller/ellipsoid projection kernels, parity-matched to frozen
//! `mc_m5r.py` (`ellipsoid`, `lfrac_ext`, `coord_range`, `maha`).
//!
//! The frozen Python programs remain the historical estimator of record.

use crate::pstt::diagnostics::linear_quantile;
use nalgebra::{DMatrix, DVector};

/// Explicit ridge contracts observed in the frozen implementations.
/// They must never be merged into a single rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RidgePolicy {
    /// M5-R / M5-S: `Sigma + 1e-12 * I`.
    M5rs,
    /// Standalone M6 (`build_m6_public.py`): `Sigma + 1e-9 * I`.
    StandaloneM6,
    /// Panel Stage 4: `Sigma_kk + 1e-12 * max(1, Sigma_kk)` on the diagonal.
    PanelStage4,
}

impl RidgePolicy {
    pub fn apply(self, sigma: &mut DMatrix<f64>) {
        let d = sigma.nrows();
        match self {
            RidgePolicy::M5rs => {
                for k in 0..d {
                    sigma[(k, k)] += 1e-12;
                }
            }
            RidgePolicy::StandaloneM6 => {
                for k in 0..d {
                    sigma[(k, k)] += 1e-9;
                }
            }
            RidgePolicy::PanelStage4 => {
                for k in 0..d {
                    let s = sigma[(k, k)];
                    sigma[(k, k)] = s + 1e-12 * s.max(1.0);
                }
            }
        }
    }
}

/// Typed projection result. Reasons for abstention are never erased.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinearFractional {
    Projected {
        lo: f64,
        hi: f64,
    },
    /// Denominator confidence set contains zero (map unbounded).
    DenominatorTouchesZero,
    NonFiniteInput,
}

impl LinearFractional {
    pub fn interval(self) -> Option<(f64, f64)> {
        match self {
            LinearFractional::Projected { lo, hi } => Some((lo, hi)),
            _ => None,
        }
    }
}

fn quad_form(a: &DVector<f64>, m: &DMatrix<f64>, b: &DVector<f64>) -> f64 {
    (a.transpose() * m * b)[(0, 0)]
}

/// Exact `[min, max]` of `(n0 + nvec.theta) / (d0 + dvec.theta)` over the
/// ellipsoid `{(theta-th)' Sigma^{-1} (theta-th) <= c}`.
///
/// Mirrors `mc_m5r.lfrac_ext` exactly, including the inclusive
/// denominator-zero check and the discriminant clamp to zero.
pub fn lfrac_ext(
    nvec: &[f64],
    n0: f64,
    dvec: &[f64],
    d0: f64,
    theta: &DVector<f64>,
    sigma: &DMatrix<f64>,
    c: f64,
) -> LinearFractional {
    if !nvec.iter().chain(dvec.iter()).all(|x| x.is_finite())
        || !n0.is_finite()
        || !d0.is_finite()
        || !c.is_finite()
        || theta.iter().any(|x| !x.is_finite())
        || sigma.iter().any(|x| !x.is_finite())
    {
        return LinearFractional::NonFiniteInput;
    }
    let n = DVector::from_column_slice(nvec);
    let d = DVector::from_column_slice(dvec);
    let n_hat = n0 + n.dot(theta);
    let d_hat = d0 + d.dot(theta);
    let s_nn = quad_form(&n, sigma, &n);
    let s_dd = quad_form(&d, sigma, &d);
    let s_nd = quad_form(&n, sigma, &d);
    let dspread = (c * s_dd).sqrt();
    if d_hat - dspread <= 0.0 && 0.0 <= d_hat + dspread {
        return LinearFractional::DenominatorTouchesZero;
    }
    let a = d_hat * d_hat - c * s_dd;
    let b = -2.0 * (n_hat * d_hat - c * s_nd);
    let cc = n_hat * n_hat - c * s_nn;
    let disc = (b * b - 4.0 * a * cc).max(0.0);
    let s = disc.sqrt();
    let g1 = (-b - s) / (2.0 * a);
    let g2 = (-b + s) / (2.0 * a);
    if !g1.is_finite() || !g2.is_finite() {
        return LinearFractional::NonFiniteInput;
    }
    LinearFractional::Projected {
        lo: g1.min(g2),
        hi: g1.max(g2),
    }
}

/// Coordinate range `th[k] +- sqrt(c * Sigma[k,k])` (frozen `coord_range`).
pub fn coord_range(k: usize, theta: &DVector<f64>, sigma: &DMatrix<f64>, c: f64) -> (f64, f64) {
    let half = (c * sigma[(k, k)]).sqrt();
    (theta[k] - half, theta[k] + half)
}

/// Mahalanobis distance `(x - th)' Sinv (x - th)` (frozen `maha`).
pub fn mahalanobis(x: &DVector<f64>, theta: &DVector<f64>, sigma_inv: &DMatrix<f64>) -> f64 {
    let diff = x - theta;
    quad_form(&diff, sigma_inv, &diff)
}

/// Sample covariance of bootstrap draws with `B-1` normalization
/// (NumPy `np.cov(star, rowvar=False)` default).
pub fn covariance_from_draws(star: &DMatrix<f64>) -> DMatrix<f64> {
    let b = star.nrows();
    let d = star.ncols();
    assert!(b >= 2, "need at least two draws for sample covariance");
    let mut means = vec![0.0; d];
    for (j, m) in means.iter_mut().enumerate() {
        *m = star.column(j).sum() / b as f64;
    }
    let mut cov = DMatrix::zeros(d, d);
    for i in 0..d {
        for j in i..d {
            let mut acc = 0.0;
            for r in 0..b {
                acc += (star[(r, i)] - means[i]) * (star[(r, j)] - means[j]);
            }
            let v = acc / (b as f64 - 1.0);
            cov[(i, j)] = v;
            cov[(j, i)] = v;
        }
    }
    cov
}

#[derive(Debug, Clone)]
pub struct EllipsoidRegion {
    pub theta: DVector<f64>,
    pub sigma: DMatrix<f64>,
    pub sigma_inv: DMatrix<f64>,
    pub c_alpha: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EllipsoidFailure {
    SingularCovariance,
    NonFiniteInput,
}

/// Build the covariance ellipsoid from bootstrap draws around `theta`,
/// applying an explicit ridge policy and the NumPy-default linear-quantile
/// Mahalanobis radius at `nominal`. Mirrors frozen `ellipsoid` / stage-2
/// standalone construction (which passes `theta` = observed sums).
pub fn ellipsoid_from_draws(
    star: &DMatrix<f64>,
    theta: &DVector<f64>,
    ridge: RidgePolicy,
    nominal: f64,
) -> Result<EllipsoidRegion, EllipsoidFailure> {
    if star.iter().any(|x| !x.is_finite()) || theta.iter().any(|x| !x.is_finite()) {
        return Err(EllipsoidFailure::NonFiniteInput);
    }
    let mut sigma = covariance_from_draws(star);
    ridge.apply(&mut sigma);
    let sigma_inv = sigma
        .clone()
        .try_inverse()
        .ok_or(EllipsoidFailure::SingularCovariance)?;
    let b = star.nrows();
    let mut d2 = Vec::with_capacity(b);
    for r in 0..b {
        let row = DVector::from_iterator(star.ncols(), star.row(r).iter().copied());
        d2.push(mahalanobis(&row, theta, &sigma_inv));
    }
    d2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let c_alpha = linear_quantile(&d2, nominal).ok_or(EllipsoidFailure::NonFiniteInput)?;
    Ok(EllipsoidRegion {
        theta: theta.clone(),
        sigma,
        sigma_inv,
        c_alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag3(a: f64, b: f64, c: f64) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_column_slice(&[a, b, c]))
    }

    #[test]
    fn constant_numerator_and_denominator() {
        let th = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        let sig = diag3(0.01, 0.01, 0.01);
        // 5 / 2 with zero-vector coefficients: constant ratio.
        let r = lfrac_ext(&[0.0, 0.0, 0.0], 5.0, &[0.0, 0.0, 0.0], 2.0, &th, &sig, 1.0);
        let (lo, hi) = r.interval().unwrap();
        assert!((lo - 2.5).abs() < 1e-12 && (hi - 2.5).abs() < 1e-12);
    }

    #[test]
    fn one_dimensional_rational_endpoints() {
        // theta scalar=4, var=1, c=1 -> theta in [3,5]; ratio theta/2 -> [1.5, 2.5]
        let th = DVector::from_column_slice(&[4.0]);
        let sig = DMatrix::from_element(1, 1, 1.0);
        let r = lfrac_ext(&[1.0], 0.0, &[0.0], 2.0, &th, &sig, 1.0);
        let (lo, hi) = r.interval().unwrap();
        assert!((lo - 1.5).abs() < 1e-9);
        assert!((hi - 2.5).abs() < 1e-9);
    }

    #[test]
    fn numerator_proportional_to_denominator() {
        // N = 3*D -> ratio constant 3 regardless of ellipsoid.
        let th = DVector::from_column_slice(&[2.0, 5.0]);
        let sig = DMatrix::from_diagonal(&DVector::from_column_slice(&[0.3, 0.7]));
        let r = lfrac_ext(&[3.0, 3.0], 0.0, &[1.0, 1.0], 0.0, &th, &sig, 2.0);
        let (lo, hi) = r.interval().unwrap();
        assert!((lo - 3.0).abs() < 1e-9 && (hi - 3.0).abs() < 1e-9);
    }

    #[test]
    fn denominator_touching_zero_is_typed() {
        // D_hat = 1, spread = sqrt(c*sDD) = 1 -> inclusive zero touch.
        let th = DVector::from_column_slice(&[1.0]);
        let sig = DMatrix::from_element(1, 1, 1.0);
        let r = lfrac_ext(&[1.0], 0.0, &[1.0], 0.0, &th, &sig, 1.0);
        assert_eq!(r, LinearFractional::DenominatorTouchesZero);
        // Slightly inside: strict positivity restores projection.
        let r2 = lfrac_ext(&[1.0], 0.0, &[1.0], 0.0, &th, &sig, 1.0 - 1e-9);
        assert!(r2.interval().is_some());
    }

    #[test]
    fn ridge_policies_are_distinct() {
        let base = DMatrix::from_diagonal(&DVector::from_column_slice(&[4.0, 0.0]));
        let mut a = base.clone();
        RidgePolicy::M5rs.apply(&mut a);
        assert!((a[(0, 0)] - (4.0 + 1e-12)).abs() < 1e-18);
        let mut b = base.clone();
        RidgePolicy::StandaloneM6.apply(&mut b);
        assert!((b[(0, 0)] - (4.0 + 1e-9)).abs() < 1e-15);
        let mut c = base.clone();
        RidgePolicy::PanelStage4.apply(&mut c);
        assert!((c[(0, 0)] - (4.0 + 1e-12 * 4.0)).abs() < 1e-18);
        assert!((c[(1, 1)] - 1e-12).abs() < 1e-18); // max(1, 0) branch
    }

    #[test]
    fn covariance_uses_b_minus_one() {
        let star = DMatrix::from_row_slice(2, 1, &[0.0, 2.0]);
        let cov = covariance_from_draws(&star);
        // mean 1, sq dev 1+1=2, /(2-1)=2
        assert!((cov[(0, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ellipsoid_radius_linear_quantile() {
        // Draws on a line around theta=0 -> D2 known; check 0.95 interpolation.
        let star = DMatrix::from_row_slice(5, 1, &[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let th = DVector::from_column_slice(&[0.0]);
        let e = ellipsoid_from_draws(&star, &th, RidgePolicy::M5rs, 0.95).unwrap();
        // Sigma = 2.5 (+ridge); D2 sorted = [0, .4, .4, 1.6, 1.6]; linear q95 between
        // index 3.8: 1.6
        assert!((e.c_alpha - 1.6).abs() < 1e-9);
    }
}
