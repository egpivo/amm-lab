//! Two-way fixed-effect residualization by alternating projections (iterative
//! demeaning). Absorbs pool and week effects from the outcome and the regressors so the
//! event-study OLS runs only on the small residualized design (Frisch--Waugh--Lovell),
//! never materializing a dense dummy matrix over thousands of pools.

/// Subtract weighted group means of `col` in place; return the largest absolute group mean
/// removed (the alternating-projections convergence signal). With unit weights this is the
/// ordinary within transform.
fn demean(col: &mut [f64], w: &[f64], group: &[usize], n_groups: usize) -> f64 {
    let mut sum = vec![0.0f64; n_groups];
    let mut wsum = vec![0.0f64; n_groups];
    for (i, &g) in group.iter().enumerate() {
        sum[g] += w[i] * col[i];
        wsum[g] += w[i];
    }
    let mut max_abs = 0.0f64;
    for g in 0..n_groups {
        if wsum[g] > 0.0 {
            sum[g] /= wsum[g];
            max_abs = max_abs.max(sum[g].abs());
        }
    }
    for (i, &g) in group.iter().enumerate() {
        col[i] -= sum[g];
    }
    max_abs
}

/// Convergence status of the alternating-projection residualization, aggregated over all
/// columns. `converged` is false if any column still had a group mean above `tol` at
/// `max_iter`; downstream code should refuse or flag a non-converged absorption.
#[derive(Debug, Clone, Copy)]
pub struct FeStatus {
    pub converged: bool,
    pub max_iters_used: usize,
    pub max_residual_group_mean: f64,
}

pub struct TwoWayFeSpec<'a> {
    pub w: &'a [f64],
    pub pool: &'a [usize],
    pub n_pool: usize,
    pub week: &'a [usize],
    pub n_week: usize,
    pub tol: f64,
    pub max_iter: usize,
}

/// Weighted two-way FE residualization by alternating projections. `w` are frequency
/// weights (matched-control multiplicity; treated units carry weight 1). `pool`/`week`
/// are 0-based group ids. Iterates weighted demean(pool) then demean(week) until the
/// largest group mean removed in a pass is below `tol`, or `max_iter` passes elapse.
/// Returns the aggregate convergence status.
pub fn residualize_twoway_weighted(cols: &mut [Vec<f64>], spec: TwoWayFeSpec<'_>) -> FeStatus {
    let mut converged = true;
    let mut max_iters_used = 0usize;
    let mut worst = 0.0f64;
    for col in cols.iter_mut() {
        let mut last = f64::INFINITY;
        let mut iters = 0usize;
        for it in 0..spec.max_iter {
            let a = demean(col, spec.w, spec.pool, spec.n_pool);
            let b = demean(col, spec.w, spec.week, spec.n_week);
            last = a.max(b);
            iters = it + 1;
            if last < spec.tol {
                break;
            }
        }
        max_iters_used = max_iters_used.max(iters);
        worst = worst.max(last);
        if last >= spec.tol {
            converged = false;
        }
    }
    FeStatus {
        converged,
        max_iters_used,
        max_residual_group_mean: worst,
    }
}

/// Unweighted two-way FE residualization (unit weights).
pub fn residualize_twoway(
    cols: &mut [Vec<f64>],
    pool: &[usize],
    n_pool: usize,
    week: &[usize],
    n_week: usize,
    tol: f64,
    max_iter: usize,
) {
    let w = vec![1.0f64; pool.len()];
    let _ = residualize_twoway_weighted(
        cols,
        TwoWayFeSpec {
            w: &w,
            pool,
            n_pool,
            week,
            n_week,
            tol,
            max_iter,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_pool_and_week_means() {
        // 3 pools x 3 weeks, y = pool_effect + week_effect + noise-free signal
        let pool_eff = [10.0, -5.0, 2.0];
        let week_eff = [1.0, 4.0, -3.0];
        let mut y = Vec::new();
        let mut pool = Vec::new();
        let mut week = Vec::new();
        for (p, pool_value) in pool_eff.iter().enumerate() {
            for (w, week_value) in week_eff.iter().enumerate() {
                y.push(*pool_value + *week_value);
                pool.push(p);
                week.push(w);
            }
        }
        let mut cols = vec![y];
        residualize_twoway(&mut cols, &pool, 3, &week, 3, 1e-12, 1000);
        // a pure two-way additive signal must residualize to ~0
        for v in &cols[0] {
            assert!(v.abs() < 1e-9, "residual not zero: {v}");
        }
    }

    #[test]
    fn preserves_within_variation() {
        // add an idiosyncratic component orthogonal to FE: it must survive
        let mut y = vec![];
        let mut pool = vec![];
        let mut week = vec![];
        let idio = [0.5, -0.5, 0.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0];
        let mut k = 0;
        for p in 0..3 {
            for w in 0..3 {
                y.push(100.0 * p as f64 + 7.0 * w as f64 + idio[k]);
                pool.push(p);
                week.push(w);
                k += 1;
            }
        }
        let mut cols = vec![y];
        residualize_twoway(&mut cols, &pool, 3, &week, 3, 1e-12, 1000);
        let ss: f64 = cols[0].iter().map(|v| v * v).sum();
        assert!(ss > 1e-6, "within variation wrongly removed");
    }
}
