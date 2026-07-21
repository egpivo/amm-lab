//! Target-moment calibration of the latent fundamental arrival hazard
//! (Poisson-arrival, round 21). For a given market cell (arb speed, sigma, z-sized
//! demand) the pooled Poisson hazard is tuned on TRAINING seeds so the
//! STATIC BASELINE's total realized AMM swap count (arb + fundamental
//! fills, i.e. primitive events with nonzero delta) matches the
//! tier-specific activity target. The identical arrival schedule is then
//! reused across all policies (policy-invariance is by construction:
//! arrivals come from a dedicated seeded RNG).

use crate::campbell::fee_policy::FixedFeePolicy;
use crate::campbell::gbm::generate_gbm;
use crate::campbell::simulation::{EventKind, SimConfig, run_simulation_with_events};

/// Realized (arb-fill, total-fill) counts per week under a fixed-fee
/// static baseline, averaged over `seeds`. Fills = primitive events with
/// nonzero delta. The arb count is the two-moment calibration's second
/// target; total is the activity target (round 26).
pub fn realized_fills_per_week(
    config: &SimConfig,
    s0: f64,
    seeds: std::ops::Range<u64>,
) -> (f64, f64) {
    let n_seeds = (seeds.end - seeds.start) as f64;
    let episode_hours = config.n_steps as f64 * config.dt_hours;
    let scale = 168.0 / episode_hours / n_seeds;
    let (mut arb, mut total) = (0.0, 0.0);
    for seed in seeds {
        let mut cfg = config.clone();
        cfg.seed = seed;
        let prices = generate_gbm(cfg.n_steps, s0, cfg.mu, cfg.sigma, cfg.dt_years(), seed);
        let mut policy = FixedFeePolicy::new(cfg.amm_fee);
        let (_, events) = run_simulation_with_events(&cfg, &prices, &mut policy);
        for e in &events {
            if e.delta != 0.0 {
                total += 1.0;
                if e.kind == EventKind::Arb {
                    arb += 1.0;
                }
            }
        }
    }
    (arb * scale, total * scale)
}

/// Realized total AMM swaps per week for a config under a fixed-fee
/// static baseline, averaged over `seeds`.
pub fn realized_swaps_per_week(config: &SimConfig, s0: f64, seeds: std::ops::Range<u64>) -> f64 {
    realized_fills_per_week(config, s0, seeds).1
}

/// Bisection on the pooled fundamental hazard so the static baseline's
/// realized total swaps/week matches `target` within `tol_rel`.
/// Returns (hazard, achieved). Panics if the target is below the
/// arb-only floor (no hazard can go lower).
pub fn calibrate_pooled_hazard(
    base: &SimConfig,
    s0: f64,
    seeds: std::ops::Range<u64>,
    target: f64,
    tol_rel: f64,
) -> (f64, f64) {
    let mut floor_cfg = base.clone();
    floor_cfg.pooled_fund_arrival_rate_per_hour = Some(0.0);
    let floor = realized_swaps_per_week(&floor_cfg, s0, seeds.clone());
    assert!(
        target > floor,
        "target {target}/wk below arb-only floor {floor}/wk — hazard cannot reach it"
    );
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    loop {
        let mut cfg = base.clone();
        cfg.pooled_fund_arrival_rate_per_hour = Some(hi);
        if realized_swaps_per_week(&cfg, s0, seeds.clone()) >= target {
            break;
        }
        hi *= 2.0;
        assert!(hi < 1.0e7, "hazard search diverged");
    }
    let mut achieved = 0.0;
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let mut cfg = base.clone();
        cfg.pooled_fund_arrival_rate_per_hour = Some(mid);
        achieved = realized_swaps_per_week(&cfg, s0, seeds.clone());
        if (achieved - target).abs() <= tol_rel * target {
            return (mid, achieved);
        }
        if achieved < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi), achieved)
}

/// Result of a two-moment cell calibration (round 26).
#[derive(Debug, Clone, Copy)]
pub struct TwoMomentResult {
    /// Latent arb-opportunity hazard (per hour) — the SOLVED simulator
    /// parameter, NOT the observed compatible-fill target.
    pub lambda_arb: f64,
    /// Latent pooled fundamental-demand hazard (per hour).
    pub lambda_fund: f64,
    pub arb_achieved: f64,
    pub total_achieved: f64,
    /// False if the arb-fill target exceeds the engine ceiling (arb
    /// active every step) — a reportable boundary, not forced.
    pub arb_reachable: bool,
}

/// Two-moment calibration for one 54-cell entry (round 26). Solves for
/// the LATENT hazards (lambda_arb*, lambda_fund*) so that the static
/// baseline's realized fills match BOTH the tier activity target
/// (total) and the proxy-calibrated arb-intensity anchor (arb fills per
/// week = arb_target_per_hour * 168). Alternating bisection: inner tunes
/// lambda_fund to hit total given lambda_arb; outer tunes lambda_arb to
/// hit the arb target given lambda_fund. NOTE the six frozen anchors are
/// OBSERVED compatible-fill RATES; the returned lambda_arb is the
/// back-solved latent OPPORTUNITY hazard (they need not be equal because
/// a scheduled opportunity only fills when the gap exceeds the band).
pub fn calibrate_two_moment(
    base: &SimConfig,
    s0: f64,
    seeds: std::ops::Range<u64>,
    total_target: f64,
    arb_target_per_hour: f64,
    tol_rel: f64,
) -> TwoMomentResult {
    // At this rate p = 1 - exp(-lambda * dt) is numerically one even at
    // the 1-second clock.  It therefore represents the engine ceiling:
    // an arb opportunity is scheduled at every simulation step.
    const ARB_HAZARD_CEILING: f64 = 1.0e6;
    let arb_target = arb_target_per_hour * 168.0;

    // Inner: given lambda_arb, bisect lambda_fund to hit TOTAL and return
    // both realized moments. Arb fills are ~monotone in lambda_arb and
    // only weakly coupled through pool state, so an OUTER bisection on
    // lambda_arb wrapping this inner fit converges the arb moment.
    let eval_at = |lam_arb: f64| -> (f64, f64, f64) {
        let mut cfg = base.clone();
        cfg.arb_arrival_rate_per_hour = Some(lam_arb);
        let (lf, _) = calibrate_pooled_hazard(&cfg, s0, seeds.clone(), total_target, tol_rel);
        let mut m = base.clone();
        m.arb_arrival_rate_per_hour = Some(lam_arb);
        m.pooled_fund_arrival_rate_per_hour = Some(lf);
        let (a, t) = realized_fills_per_week(&m, s0, seeds.clone());
        (lf, a, t)
    };

    // bracket lambda_arb: lo gives arb below target, hi above.
    // Reachability is evaluated inside this joint calibration: every
    // lambda_arb evaluation first re-fits lambda_fund to the total-fill
    // target.  Only cells whose bracket reaches the scheduling ceiling
    // need the expensive ceiling evaluation.
    let mut lo = 0.0f64;
    let (mut lf_lo, _, _) = eval_at(lo);
    let mut hi = 1.0f64;
    let (mut lf_hi, mut arb_hi, mut tot_hi) = eval_at(hi);
    while arb_hi < arb_target && hi < ARB_HAZARD_CEILING {
        hi = (hi * 2.0).min(ARB_HAZARD_CEILING);
        let e = eval_at(hi);
        lf_hi = e.0;
        arb_hi = e.1;
        tot_hi = e.2;
    }
    let arb_reachable = arb_hi >= (1.0 - tol_rel) * arb_target;
    if !arb_reachable || arb_hi < arb_target {
        return TwoMomentResult {
            lambda_arb: hi,
            lambda_fund: lf_hi,
            arb_achieved: arb_hi,
            total_achieved: tot_hi,
            arb_reachable,
        };
    }
    let (mut lam_fund, mut arb_ach, mut tot_ach) = (lf_hi, arb_hi, tot_hi);
    let mut lam_arb = hi;
    let _ = lf_lo;
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        let (lf, a, t) = eval_at(mid);
        lam_arb = mid;
        lam_fund = lf;
        arb_ach = a;
        tot_ach = t;
        if (a - arb_target).abs() <= tol_rel * arb_target {
            break;
        }
        if a < arb_target {
            lo = mid;
            lf_lo = lf;
        } else {
            hi = mid;
        }
    }
    let _ = lf_lo;
    TwoMomentResult {
        lambda_arb: lam_arb,
        lambda_fund: lam_fund,
        arb_achieved: arb_ach,
        total_achieved: tot_ach,
        arb_reachable,
    }
}
