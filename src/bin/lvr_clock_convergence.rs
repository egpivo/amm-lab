//! Round-23 clock-convergence diagnostic (M3 gate 1, corrected design).
//!
//! COMMON CONTINUOUS-TIME BASE per seed (round-23 requirement): one
//! 15-second Brownian increment stream (1-minute and 5-minute prices are
//! subsampled from the SAME cumulative path), one continuous-time
//! fundamental Poisson event schedule (times, sides), and one
//! continuous-time arbitrage-opportunity schedule; the three clocks only
//! BIN the same objects. Injection bypasses internal draws.
//!
//! PRIMARY test holds latent parameters fixed: lambda_fund is calibrated
//! ONCE on the 15-second benchmark (calibration pilot seeds) and reused
//! verbatim under the 1-minute and 5-minute clocks — per-clock
//! recalibration would let the hazard absorb clock error (it is reported
//! separately as a SECONDARY operational-equivalence check, not here).
//!
//! Metric criteria (round 23): symmetric relative distance for A, B, S;
//! pool-value-normalized differences and sign checks for U; absolute
//! percentage-point distance for allocation; policy comparisons via
//! per-seed paired differences Delta U_ij with margins, not rank of
//! sample means alone.
//!
//! SEEDS: everything here uses the PILOT/DESIGN domain (seeds < 10,000),
//! permanently retired from calibration/training/validation/final.

use amm_lab::campbell::calibrate::calibrate_pooled_hazard;
use amm_lab::campbell::fee_policy::{FeePolicy, FixedFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::simulation::{
    ArrivalModel, FlowRegime, InjectedSchedules, SimConfig, run_simulation_with_injected_schedules,
};
use amm_lab::campbell::summary::summarize_events;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Poisson};

const S0: f64 = 2000.0;
const V0: f64 = 4.0e7;
const WEEK_HOURS: f64 = 168.0;
const N15: usize = 40_320; // 15-second steps in one week
const CAL_SEEDS: std::ops::Range<u64> = 500..503; // pilot domain
const EVAL_SEEDS: std::ops::Range<u64> = 1000..1020; // pilot domain (retired)

struct Clock {
    name: &'static str,
    dt_hours: f64,
    n_steps: usize,
    lag_steps: usize,
    agg: usize, // 15s increments per step
}

fn clocks() -> Vec<Clock> {
    vec![
        Clock {
            name: "5min",
            dt_hours: 1.0 / 12.0,
            n_steps: 2016,
            lag_steps: 1,
            agg: 20,
        },
        Clock {
            name: "1min",
            dt_hours: 1.0 / 60.0,
            n_steps: 10_080,
            lag_steps: 5,
            agg: 4,
        },
        Clock {
            name: "15s",
            dt_hours: 1.0 / 240.0,
            n_steps: N15,
            lag_steps: 20,
            agg: 1,
        },
    ]
}

/// Shared continuous-time base for one seed.
struct ContinuousBase {
    log_path15: Vec<f64>, // cumulative log price at 15s boundaries (len N15+1)
    fund_times: Vec<(f64, bool)>, // (hours, is_buy), sorted
    arb_times: Vec<f64>,  // hours, sorted
}

fn gen_base(seed: u64, sigma: f64, lam_fund: f64, lam_arb: f64) -> ContinuousBase {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xC10C_C10C);
    let dt15_years = (1.0 / 240.0) / (24.0 * 365.0);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut log_path15 = Vec::with_capacity(N15 + 1);
    let mut acc = 0.0f64;
    log_path15.push(acc);
    for _ in 0..N15 {
        acc +=
            -0.5 * sigma * sigma * dt15_years + sigma * dt15_years.sqrt() * normal.sample(&mut rng);
        log_path15.push(acc);
    }
    let mut fund_times: Vec<(f64, bool)> = Vec::new();
    if lam_fund > 0.0 {
        let n = Poisson::new(lam_fund * WEEK_HOURS)
            .unwrap()
            .sample(&mut rng) as usize;
        for _ in 0..n {
            fund_times.push((
                rng.gen_range(0.0..WEEK_HOURS),
                rng.gen_range(0.0f64..1.0) < 0.5,
            ));
        }
        fund_times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
    let mut arb_times: Vec<f64> = Vec::new();
    if lam_arb > 0.0 {
        let n = Poisson::new(lam_arb * WEEK_HOURS).unwrap().sample(&mut rng) as usize;
        for _ in 0..n {
            arb_times.push(rng.gen_range(0.0..WEEK_HOURS));
        }
        arb_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    ContinuousBase {
        log_path15,
        fund_times,
        arb_times,
    }
}

/// Bin the shared base under one clock.
fn bin_base(base: &ContinuousBase, ck: &Clock) -> (Vec<f64>, InjectedSchedules) {
    let prices: Vec<f64> = (0..=ck.n_steps)
        .map(|j| S0 * base.log_path15[j * ck.agg].exp())
        .collect();
    let mut fund: Vec<Vec<(f64, bool)>> = vec![Vec::new(); ck.n_steps];
    for &(t, is_buy) in &base.fund_times {
        let x = t / ck.dt_hours;
        let step = (x.floor() as usize).min(ck.n_steps - 1);
        fund[step].push((x - step as f64, is_buy));
    }
    for f in &mut fund {
        f.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
    let mut arb_active = vec![false; ck.n_steps];
    for &t in &base.arb_times {
        let step = ((t / ck.dt_hours).floor() as usize).min(ck.n_steps - 1);
        arb_active[step] = true;
    }
    (prices, InjectedSchedules { fund, arb_active })
}

fn config_for(fee: f64, sigma: f64, z: f64, ck: &Clock) -> SimConfig {
    let y0 = 1.0e4;
    let d_ref = y0 * (1.0 - (1.0f64 + 0.01).powf(-0.5));
    SimConfig {
        name: "clockconv".into(),
        description: "round-23 clock convergence (common base)".into(),
        amm_fee: fee,
        cex_fee: 0.0010,
        buy_demand: z * d_ref,
        sell_demand: z * d_ref,
        reserve_x: 2.0e7,
        reserve_y: y0,
        sigma,
        mu: 0.0,
        n_steps: ck.n_steps,
        seed: 0,
        flow_regime: FlowRegime::Normal,
        toxic_burst_prob: 0.0,
        toxic_burst_arb_scale: 1.0,
        toxic_burst_fund_scale: 1.0,
        regime_switch_period: 0,
        e1_lambda: 0.0,
        e1_fee_ref: 0.0006,
        e5_arb_prob: 1.0,
        policy_lag: ck.lag_steps,
        dt_hours: ck.dt_hours,
        pooled_fund_arrival_rate_per_hour: Some(1.0),
        buy_arrival_share: 0.5,
        arb_arrival_rate_per_hour: Some(1.0),
        lookback_hours: 20.0,
        arrival_model: ArrivalModel::Poisson,
        log_inactive_arb: false,
    }
}

fn make_policy(which: usize, fee: f64) -> Box<dyn FeePolicy> {
    match which {
        0 => Box::new(FixedFeePolicy::new(fee)),
        1 => Box::new(OracleGapFeePolicy {
            base_fee: fee,
            gap_multiplier: 2.0,
            min_fee: fee,
            max_fee: 10.0 * fee,
        }),
        _ => Box::new(OracleGapFeePolicy {
            base_fee: fee,
            gap_multiplier: 50.0,
            min_fee: fee,
            max_fee: 0.30,
        }),
    }
}

#[derive(Clone, Copy, Default)]
struct M {
    a: f64,
    b: f64,
    u: f64,
    s: f64,
    alloc: f64,
}

fn d_sym(x: f64, y: f64) -> f64 {
    2.0 * (x - y).abs() / (x.abs() + y.abs() + 1e-12)
}

fn main() {
    let strata = [("5bp", 0.0005, 41_500.0), ("30bp", 0.0030, 3_000.0)];
    let sigmas = [0.48, 0.92];
    let zs = [0.00087, 0.030];
    let arbs = [2.0, 20.0]; // PROVISIONAL physical arb-opportunity rates /hr
    let names = ["static", "gap", "defensive"];
    let n_eval = (EVAL_SEEDS.end - EVAL_SEEDS.start) as usize;

    let mut summary = Vec::new();
    for (sname, fee, target) in strata {
        for &sigma in &sigmas {
            for &z in &zs {
                for &arb in &arbs {
                    let cell = format!("{sname} s{sigma} z{z} arb{arb}");
                    // PRIMARY: calibrate lambda at the 15s benchmark only,
                    // production engine, calibration pilot seeds.
                    let ck15 = &clocks()[2];
                    let mut cal_cfg = config_for(fee, sigma, z, ck15);
                    cal_cfg.arb_arrival_rate_per_hour = Some(arb);
                    let (lam, achieved) =
                        calibrate_pooled_hazard(&cal_cfg, S0, CAL_SEEDS, target, 0.05);

                    // per clock x policy x seed U (for paired diffs) + means
                    let mut means = vec![vec![M::default(); 3]; 3];
                    let mut u_per_seed = vec![vec![vec![0.0f64; n_eval]; 3]; 3];
                    for (ci, ck) in clocks().iter().enumerate() {
                        for (si, seed) in EVAL_SEEDS.enumerate() {
                            let base = gen_base(seed, sigma, lam, arb);
                            let (prices, sched) = bin_base(&base, ck);
                            for (pi, _) in names.iter().enumerate() {
                                let mut cfg = config_for(fee, sigma, z, ck);
                                cfg.seed = seed;
                                cfg.pooled_fund_arrival_rate_per_hour = Some(lam);
                                cfg.arb_arrival_rate_per_hour = Some(arb);
                                let mut pol = make_policy(pi, fee);
                                let (records, evs) = run_simulation_with_injected_schedules(
                                    &cfg,
                                    &prices,
                                    pol.as_mut(),
                                    &sched,
                                );
                                let es = summarize_events(&evs, &records);
                                let m = &mut means[ci][pi];
                                m.a += es.a_fill / n_eval as f64;
                                m.b += es.b_fill / n_eval as f64;
                                m.u += es.u_lp_rel / n_eval as f64;
                                m.s += es.served_fund_volume / n_eval as f64;
                                m.alloc += es.alloc_amm_share.unwrap_or(f64::NAN) / n_eval as f64;
                                u_per_seed[ci][pi][si] = es.u_lp_rel;
                            }
                        }
                    }
                    println!("\n{cell}: lam15={lam:.1}/hr (achieved {achieved:.0}/wk)");
                    for (ci, ck) in clocks().iter().enumerate() {
                        println!(
                            "  {:5} {}",
                            ck.name,
                            (0..3)
                                .map(|p| format!(
                                    "{}: A={:.0} B={:.2} U/V0={:.2e} S={:.1} alloc={:.3}",
                                    names[p],
                                    means[ci][p].a,
                                    means[ci][p].b,
                                    means[ci][p].u / V0,
                                    means[ci][p].s,
                                    means[ci][p].alloc
                                ))
                                .collect::<Vec<_>>()
                                .join(" | ")
                        );
                    }
                    // verdicts vs 15s
                    for ci in 0..2 {
                        let mut worst_dsym = 0.0f64;
                        let mut worst_dupv0 = 0.0f64;
                        let mut worst_alloc_pp = 0.0f64;
                        for (mean_ci, mean_bench) in means[ci].iter().zip(means[2].iter()) {
                            worst_dsym = worst_dsym
                                .max(d_sym(mean_ci.a, mean_bench.a))
                                .max(d_sym(mean_ci.b, mean_bench.b))
                                .max(d_sym(mean_ci.s, mean_bench.s));
                            worst_dupv0 = worst_dupv0.max((mean_ci.u - mean_bench.u).abs() / V0);
                            worst_alloc_pp =
                                worst_alloc_pp.max((mean_ci.alloc - mean_bench.alloc).abs());
                        }
                        // paired pairwise sign agreement with margins
                        let mut sign_mismatch = 0usize;
                        let mut min_margin = f64::INFINITY;
                        for i in 0..3 {
                            for j in (i + 1)..3 {
                                let d_bench: f64 = (0..n_eval)
                                    .map(|s| u_per_seed[2][i][s] - u_per_seed[2][j][s])
                                    .sum::<f64>()
                                    / n_eval as f64;
                                let d_this: f64 = (0..n_eval)
                                    .map(|s| u_per_seed[ci][i][s] - u_per_seed[ci][j][s])
                                    .sum::<f64>()
                                    / n_eval as f64;
                                min_margin = min_margin.min(d_bench.abs() / V0);
                                if d_bench.signum() != d_this.signum() {
                                    sign_mismatch += 1;
                                }
                            }
                        }
                        println!(
                            "  VERDICT {} vs 15s: dsym(A,B,S)max={worst_dsym:.4} |dU|/V0max={worst_dupv0:.2e} alloc_pp={worst_alloc_pp:.4} pair_sign_mismatch={sign_mismatch} min_margin/V0={min_margin:.2e}",
                            clocks()[ci].name
                        );
                        summary.push((clocks()[ci].name, worst_dsym, worst_dupv0, sign_mismatch));
                    }
                }
            }
        }
    }
    println!("\n== overall ==");
    for cname in ["5min", "1min"] {
        let rows: Vec<_> = summary.iter().filter(|r| r.0 == cname).collect();
        let wd = rows.iter().map(|r| r.1).fold(0.0f64, f64::max);
        let wu = rows.iter().map(|r| r.2).fold(0.0f64, f64::max);
        let sm: usize = rows.iter().map(|r| r.3).sum();
        println!(
            "{cname}: worst dsym={wd:.4} worst |dU|/V0={wu:.2e} total pair-sign mismatches={sm}"
        );
    }
}
