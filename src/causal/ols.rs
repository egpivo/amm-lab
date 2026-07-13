//! OLS on the residualized design, token-pair cluster-robust variance, and a wild
//! cluster bootstrap (Rademacher weights, bootstrap-$t$). Phase 1: the primary
//! estimator and core inference only; Honest-DiD / robustness-value / entropy-balancing
//! modules are deliberately deferred and added separately once their numerical checks
//! are explicit.

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn is_positive(v: f64) -> bool {
    matches!(v.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater))
}

pub struct OlsFit {
    pub beta: DVector<f64>,
    pub resid: DVector<f64>,
    pub xtx_inv: DMatrix<f64>,
    pub n: usize,
    pub k: usize,
    /// numerical rank of the design (singular values above tolerance)
    pub rank: usize,
    /// condition number s_max / s_min of the design
    pub cond: f64,
}

#[derive(Debug)]
pub enum OlsError {
    /// design rank `rank` < columns `k` (near-collinear / empty bins); `cond` reported
    RankDeficient {
        rank: usize,
        k: usize,
        cond: f64,
    },
    /// N <= K: no residual degrees of freedom
    NoResidualDof {
        n: usize,
        k: usize,
    },
    Svd,
}

/// OLS by SVD (numerically stable for near-collinear event-time / baseline-time designs).
/// Returns rank and condition-number diagnostics and refuses a rank-deficient design
/// rather than returning surface-valid coefficients.
pub fn ols(x: &DMatrix<f64>, y: &DVector<f64>) -> Result<OlsFit, OlsError> {
    let (n, k) = (x.nrows(), x.ncols());
    if n <= k {
        return Err(OlsError::NoResidualDof { n, k });
    }
    let svd = x.clone().svd(true, true);
    let s = &svd.singular_values;
    let smax = s.iter().cloned().fold(0.0f64, f64::max);
    let smin = s.iter().cloned().fold(f64::INFINITY, f64::min);
    let tol = smax * (n.max(k) as f64) * f64::EPSILON;
    let rank = s.iter().filter(|&&v| v > tol).count();
    let cond = if smin > 0.0 {
        smax / smin
    } else {
        f64::INFINITY
    };
    if rank < k {
        return Err(OlsError::RankDeficient { rank, k, cond });
    }
    let vt = svd.v_t.as_ref().ok_or(OlsError::Svd)?;
    let v = vt.transpose(); // k x k
    // (X'X)^{-1} = V diag(1/s^2) V^T ; beta = V diag(1/s) U^T y
    let mut inv_s2 = DVector::<f64>::zeros(k);
    for j in 0..k {
        inv_s2[j] = 1.0 / (s[j] * s[j]);
    }
    let xtx_inv = &v * DMatrix::from_diagonal(&inv_s2) * &v.transpose();
    let beta = svd.solve(y, tol).map_err(|_| OlsError::Svd)?;
    let resid = y - x * &beta;
    Ok(OlsFit {
        n,
        k,
        beta,
        resid,
        xtx_inv,
        rank,
        cond,
    })
}

/// Cluster-robust sandwich variance with the standard CR1 small-sample factor
/// `c = G/(G-1) * (N-1)/(N-K)`. `n_eff` is the effective (frequency-weighted) observation
/// count that matches the weighting story: with unit weights it is the row count, and with
/// matched-control multiplicity it is `sum(weights)` (the represented sample size). The
/// score `meat` is already weighted through the `sqrt(w)`-scaled design and residuals.
/// Caller must ensure `n_clusters >= 2` and `n_eff > K` (enforced upstream in `run`).
pub fn cluster_vcov(
    x: &DMatrix<f64>,
    resid: &DVector<f64>,
    xtx_inv: &DMatrix<f64>,
    cluster: &[usize],
    n_clusters: usize,
    n_eff: f64,
) -> DMatrix<f64> {
    let k = x.ncols();
    let mut scores = vec![DVector::<f64>::zeros(k); n_clusters];
    for i in 0..x.nrows() {
        let xi = x.row(i).transpose(); // k x 1
        scores[cluster[i]] += xi * resid[i];
    }
    let mut meat = DMatrix::<f64>::zeros(k, k);
    for s in &scores {
        meat += s * s.transpose();
    }
    let g = n_clusters as f64;
    let kk = k as f64;
    let c = (g / (g - 1.0)) * ((n_eff - 1.0) / (n_eff - kk));
    xtx_inv * (meat * c) * xtx_inv
}

pub struct CoefInference {
    pub beta: f64,
    pub se: f64,
    pub t: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub boot_ci_lo: f64,
    pub boot_ci_hi: f64,
    /// unrestricted (WCU) wild-cluster bootstrap-$t$ p-value for H0: coef = 0
    pub boot_p: f64,
    /// restricted (WCR) wild-cluster bootstrap-$t$ p-value for H0: coef = 0 -- the
    /// Cameron-Gelbach-Miller default with few clusters. NaN when not computed / non-inferable.
    pub wcr_p: f64,
}

pub struct BootstrapSpec<'a> {
    pub cluster: &'a [usize],
    pub n_clusters: usize,
    pub n_eff: f64,
    pub n_boot: usize,
    pub alpha: f64,
    pub seed: u64,
}

/// Wild cluster bootstrap-$t$ (Rademacher weights) for every coefficient. Uses the fixed
/// residualized design `x`; for each replication it reweights cluster residuals by a
/// shared sign, re-estimates, and forms a cluster-robust $t$. The seed is caller-supplied
/// and recorded in the estimator manifest so the interval is exactly reproducible.
pub fn wild_cluster_bootstrap(
    x: &DMatrix<f64>,
    fit: &OlsFit,
    spec: BootstrapSpec<'_>,
) -> Vec<CoefInference> {
    let k = fit.k;
    let v0 = cluster_vcov(
        x,
        &fit.resid,
        &fit.xtx_inv,
        spec.cluster,
        spec.n_clusters,
        spec.n_eff,
    );
    let se0: Vec<f64> = (0..k).map(|j| v0[(j, j)].max(0.0).sqrt()).collect();
    let t0: Vec<f64> = (0..k)
        .map(|j| {
            if is_positive(se0[j]) {
                fit.beta[j] / se0[j]
            } else {
                f64::NAN
            }
        })
        .collect();
    let fitted = x * &fit.beta;

    let mut abs_t: Vec<Vec<f64>> = vec![Vec::with_capacity(spec.n_boot); k]; // |t*| per coef
    let mut rng = StdRng::seed_from_u64(spec.seed);
    let xt = x.transpose();
    for _ in 0..spec.n_boot {
        let signs: Vec<f64> = (0..spec.n_clusters)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
            .collect();
        let mut ystar = DVector::<f64>::zeros(fit.n);
        for i in 0..fit.n {
            ystar[i] = fitted[i] + signs[spec.cluster[i]] * fit.resid[i];
        }
        let beta_star = &fit.xtx_inv * (&xt * &ystar);
        let resid_star = &ystar - x * &beta_star;
        let vstar = cluster_vcov(
            x,
            &resid_star,
            &fit.xtx_inv,
            spec.cluster,
            spec.n_clusters,
            spec.n_eff,
        );
        for j in 0..k {
            let se = vstar[(j, j)].max(0.0).sqrt();
            let tj = if is_positive(se) {
                (beta_star[j] - fit.beta[j]) / se
            } else {
                0.0
            };
            abs_t[j].push(tj.abs());
        }
    }

    (0..k)
        .map(|j| {
            if !is_positive(se0[j]) {
                // zero/degenerate variance (e.g. a near-empty bin): report non-inferable
                // rather than an inf/NaN that looks like a valid test
                return CoefInference {
                    beta: fit.beta[j],
                    se: se0[j],
                    t: f64::NAN,
                    ci_lo: f64::NAN,
                    ci_hi: f64::NAN,
                    boot_ci_lo: f64::NAN,
                    boot_ci_hi: f64::NAN,
                    boot_p: f64::NAN,
                    wcr_p: f64::NAN,
                };
            }
            let mut ts: Vec<f64> = abs_t[j].iter().cloned().filter(|v| v.is_finite()).collect();
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = if ts.is_empty() {
                f64::NAN
            } else {
                let pos = ((1.0 - spec.alpha) * (ts.len() as f64 - 1.0)).round() as usize;
                ts[pos.min(ts.len() - 1)]
            };
            let ge = abs_t[j]
                .iter()
                .filter(|v| v.is_finite() && **v >= t0[j].abs())
                .count();
            CoefInference {
                beta: fit.beta[j],
                se: se0[j],
                t: t0[j],
                ci_lo: fit.beta[j] - 1.959964 * se0[j],
                ci_hi: fit.beta[j] + 1.959964 * se0[j],
                boot_ci_lo: fit.beta[j] - q * se0[j],
                boot_ci_hi: fit.beta[j] + q * se0[j],
                boot_p: ge as f64 / spec.n_boot as f64,
                wcr_p: f64::NAN, // filled by `wcr_pvalues` in `run`
            }
        })
        .collect()
}

/// Restricted wild cluster bootstrap-$t$ p-values (Cameron-Gelbach-Miller "WCR"), the
/// recommended default with few clusters. For each coefficient `j` it imposes H0: coef = 0
/// by refitting the design with column `j` removed, resamples the *restricted* residuals
/// with cluster-level Rademacher signs, refits the UNRESTRICTED model on each bootstrap
/// outcome, and compares the bootstrap $|t_j|$ (centred at 0, since the null is imposed) to
/// the observed $|t_j|$. Returns one p-value per coefficient (NaN where the coefficient is
/// non-inferable or the restricted refit is rank-deficient). Per-coefficient seeds are
/// derived from `seed` so the whole vector is reproducible.
pub fn wcr_pvalues(
    x: &DMatrix<f64>,
    fit: &OlsFit,
    cluster: &[usize],
    n_clusters: usize,
    n_eff: f64,
    n_boot: usize,
    seed: u64,
) -> Vec<f64> {
    let (n, k) = (fit.n, fit.k);
    let v0 = cluster_vcov(x, &fit.resid, &fit.xtx_inv, cluster, n_clusters, n_eff);
    let se0: Vec<f64> = (0..k).map(|j| v0[(j, j)].max(0.0).sqrt()).collect();
    // observed unrestricted t under H0: coef_j = 0
    let t0_abs: Vec<f64> = (0..k)
        .map(|j| {
            if is_positive(se0[j]) {
                (fit.beta[j] / se0[j]).abs()
            } else {
                f64::NAN
            }
        })
        .collect();
    // original residualized outcome, reconstructed from the unrestricted fit
    let y = x * &fit.beta + &fit.resid;
    let xt = x.transpose();

    let mut out = vec![f64::NAN; k];
    for j in 0..k {
        if !is_positive(se0[j]) {
            continue; // non-inferable coefficient
        }
        // restricted design with column j removed -> restricted fit under the null
        let keep: Vec<usize> = (0..k).filter(|&c| c != j).collect();
        let xr = x.select_columns(&keep);
        let fitr = match ols(&xr, &y) {
            Ok(f) => f,
            Err(_) => continue, // restricted design rank-deficient -> leave NaN
        };
        let fitted_r = &xr * &fitr.beta;
        let resid_r = &y - &fitted_r;

        // per-coefficient reproducible stream
        let mut rng = StdRng::seed_from_u64(seed ^ (j as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let mut ge = 0usize;
        let mut used = 0usize;
        for _ in 0..n_boot {
            let signs: Vec<f64> = (0..n_clusters)
                .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
                .collect();
            let mut ystar = DVector::<f64>::zeros(n);
            for i in 0..n {
                ystar[i] = fitted_r[i] + signs[cluster[i]] * resid_r[i];
            }
            // refit the UNRESTRICTED model on the null-imposed bootstrap outcome
            let beta_star = &fit.xtx_inv * (&xt * &ystar);
            let resid_star = &ystar - x * &beta_star;
            let vstar = cluster_vcov(x, &resid_star, &fit.xtx_inv, cluster, n_clusters, n_eff);
            let se = vstar[(j, j)].max(0.0).sqrt();
            if is_positive(se) {
                let t_star = (beta_star[j] / se).abs(); // centred at 0: null imposed
                if t_star.is_finite() {
                    used += 1;
                    if t_star >= t0_abs[j] {
                        ge += 1;
                    }
                }
            }
        }
        out[j] = if used > 0 {
            ge as f64 / used as f64
        } else {
            f64::NAN
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(rows: &[(f64, f64)]) -> (DMatrix<f64>, DVector<f64>) {
        // columns: intercept, x
        let n = rows.len();
        let mut x = DMatrix::zeros(n, 2);
        let mut y = DVector::zeros(n);
        for (i, &(xi, yi)) in rows.iter().enumerate() {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = xi;
            y[i] = yi;
        }
        (x, y)
    }

    #[test]
    fn ols_recovers_known_coefficients() {
        // y = 2 + 3x exactly
        let rows: Vec<(f64, f64)> = (0..20).map(|i| (i as f64, 2.0 + 3.0 * i as f64)).collect();
        let (x, y) = design(&rows);
        let fit = ols(&x, &y).unwrap();
        assert!(
            (fit.beta[0] - 2.0).abs() < 1e-9,
            "intercept {}",
            fit.beta[0]
        );
        assert!((fit.beta[1] - 3.0).abs() < 1e-9, "slope {}", fit.beta[1]);
    }

    #[test]
    fn cluster_vcov_is_finite_and_psd_diagonal() {
        let rows: Vec<(f64, f64)> = (0..40)
            .map(|i| (i as f64, 1.0 + 0.5 * i as f64 + ((i % 3) as f64 - 1.0)))
            .collect();
        let (x, y) = design(&rows);
        let fit = ols(&x, &y).unwrap();
        let cl: Vec<usize> = (0..40).map(|i| i % 5).collect();
        let v = cluster_vcov(&x, &fit.resid, &fit.xtx_inv, &cl, 5, 40.0);
        for j in 0..2 {
            assert!(v[(j, j)].is_finite() && v[(j, j)] >= 0.0);
        }
    }

    #[test]
    fn wcr_is_deterministic_and_detects_strong_signal() {
        // strong slope relative to tiny cluster noise -> restricted-null bootstrap |t*| almost
        // never exceeds the observed |t| -> WCR p ~ 0 for the slope.
        let rows: Vec<(f64, f64)> = (0..80)
            .map(|i| (i as f64, 5.0 * i as f64 + ((i % 8) as f64 - 3.5) * 0.01))
            .collect();
        let (x, y) = design(&rows);
        let fit = ols(&x, &y).unwrap();
        let cl: Vec<usize> = (0..80).map(|i| i % 8).collect();
        let p1 = wcr_pvalues(&x, &fit, &cl, 8, 80.0, 300, 7);
        let p2 = wcr_pvalues(&x, &fit, &cl, 8, 80.0, 300, 7);
        assert_eq!(p1.len(), 2);
        // reproducible under the same seed
        assert!(
            (p1[1] - p2[1]).abs() < 1e-12,
            "wcr not deterministic: {} vs {}",
            p1[1],
            p2[1]
        );
        // valid probability
        assert!(
            (0.0..=1.0).contains(&p1[1]),
            "wcr p out of range: {}",
            p1[1]
        );
        // strong signal -> small p
        assert!(
            p1[1] < 0.05,
            "expected tiny WCR p for strong slope, got {}",
            p1[1]
        );
    }

    #[test]
    fn wild_bootstrap_is_deterministic_under_seed() {
        let rows: Vec<(f64, f64)> = (0..60)
            .map(|i| (i as f64, 1.0 + 0.5 * i as f64 + ((i % 4) as f64 - 1.5)))
            .collect();
        let (x, y) = design(&rows);
        let fit = ols(&x, &y).unwrap();
        let cl: Vec<usize> = (0..60).map(|i| i % 6).collect();
        let a = wild_cluster_bootstrap(
            &x,
            &fit,
            BootstrapSpec {
                cluster: &cl,
                n_clusters: 6,
                n_eff: 60.0,
                n_boot: 200,
                alpha: 0.05,
                seed: 42,
            },
        );
        let fit2 = ols(&x, &y).unwrap();
        let b = wild_cluster_bootstrap(
            &x,
            &fit2,
            BootstrapSpec {
                cluster: &cl,
                n_clusters: 6,
                n_eff: 60.0,
                n_boot: 200,
                alpha: 0.05,
                seed: 42,
            },
        );
        assert!((a[1].boot_p - b[1].boot_p).abs() < 1e-12);
        assert!((a[1].boot_ci_hi - b[1].boot_ci_hi).abs() < 1e-12);
    }
}
