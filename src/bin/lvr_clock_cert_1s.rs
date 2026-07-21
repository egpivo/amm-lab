//! Round-25 ONE-SECOND certification: final clock check. The 15-second
//! clock failed its preregistered certification (round 24, branch (a):
//! no threshold amendments); this binary certifies the 5-second clock
//! against a 1-second benchmark on the same four worst-case cells.
//!
//! STOPPING RULE (frozen with the run): all four cells pass => primary
//! clock = 5 seconds; any static/gap level failure or non-near-tie sign
//! reversal => primary clock = 1 second, labeled "the finest
//! preregistered numerical resolution" (not proven continuous-time
//! convergence). No further threshold amendments and no sub-second
//! certification either way.
//!
//! Four worst-case cells (both strata x sigma 0.92 x z 0.03 x
//! slow/fast arb), three policies. One shared 5-SECOND continuous-time
//! base per seed (Brownian increments, fundamental event times/sides,
//! arb-opportunity times); the 15-second version aggregates 3
//! increments and re-bins the same schedules. The latent hazard is
//! FIXED at the 15-second-benchmark calibration; no per-clock
//! recalibration in the primary comparison. Physical signal staleness
//! stays 5 minutes (lag = 60 steps at 5 s, 20 steps at 15 s).
//!
//! ACCEPTANCE CRITERIA (frozen BEFORE inspection, round 24):
//! - static & gap: d_sym(A), d_sym(B), d_sym(S) <= 2%; alloc diff <=
//!   1 pp; |U_15s - U_5s|/V0 <= 5e-5 with unchanged U sign;
//! - policy pairs: paired DeltaU signs agree for non-near-ties
//!   (near tie := |DeltaU_5s|/V0 < 2e-5; flips there are disclosed,
//!   not disqualifying);
//! - defensive near-shutdown quantities: absolute fallbacks — |dA|/V0
//!   and |dU|/V0 below 0.1 bp (1e-5), service diff reported absolute,
//!   shutdown classification (S < 1% of static's S) must agree.

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
const N1: usize = 604_800; // 1-second steps per week
const CAL_SEEDS: std::ops::Range<u64> = 500..503; // pilot domain
const EVAL_SEEDS: std::ops::Range<u64> = 3000..3020; // pilot domain (fresh)

struct Clock {
    name: &'static str,
    dt_hours: f64,
    n_steps: usize,
    lag_steps: usize,
    agg: usize,
}

fn clocks() -> [Clock; 2] {
    [
        Clock {
            name: "5s",
            dt_hours: 1.0 / 720.0,
            n_steps: 120_960,
            lag_steps: 60,
            agg: 5,
        },
        Clock {
            name: "1s",
            dt_hours: 1.0 / 3600.0,
            n_steps: N1,
            lag_steps: 300,
            agg: 1,
        },
    ]
}

struct Base {
    log_path1: Vec<f64>,
    fund_times: Vec<(f64, bool)>,
    arb_times: Vec<f64>,
}

fn gen_base(seed: u64, sigma: f64, lam_fund: f64, lam_arb: f64) -> Base {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x1CE7_1CE7);
    let dt1_years = (1.0 / 3600.0) / (24.0 * 365.0);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut path = Vec::with_capacity(N1 + 1);
    let mut acc = 0.0;
    path.push(acc);
    for _ in 0..N1 {
        acc +=
            -0.5 * sigma * sigma * dt1_years + sigma * dt1_years.sqrt() * normal.sample(&mut rng);
        path.push(acc);
    }
    let mut fund = Vec::new();
    if lam_fund > 0.0 {
        let n = Poisson::new(lam_fund * WEEK_HOURS)
            .unwrap()
            .sample(&mut rng) as usize;
        for _ in 0..n {
            fund.push((
                rng.gen_range(0.0..WEEK_HOURS),
                rng.gen_range(0.0f64..1.0) < 0.5,
            ));
        }
        fund.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
    let mut arb = Vec::new();
    if lam_arb > 0.0 {
        let n = Poisson::new(lam_arb * WEEK_HOURS).unwrap().sample(&mut rng) as usize;
        for _ in 0..n {
            arb.push(rng.gen_range(0.0..WEEK_HOURS));
        }
        arb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    Base {
        log_path1: path,
        fund_times: fund,
        arb_times: arb,
    }
}

fn bin_base(base: &Base, ck: &Clock) -> (Vec<f64>, InjectedSchedules) {
    let prices: Vec<f64> = (0..=ck.n_steps)
        .map(|j| S0 * base.log_path1[j * ck.agg].exp())
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
        arb_active[((t / ck.dt_hours).floor() as usize).min(ck.n_steps - 1)] = true;
    }
    (prices, InjectedSchedules { fund, arb_active })
}

fn config_for(fee: f64, sigma: f64, z: f64, ck: &Clock) -> SimConfig {
    let y0 = 1.0e4;
    let d_ref = y0 * (1.0 - (1.0f64 + 0.01).powf(-0.5));
    SimConfig {
        name: "cert1s".into(),
        description: "round-25 1s certification".into(),
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

fn d_sym(x: f64, y: f64) -> f64 {
    2.0 * (x - y).abs() / (x.abs() + y.abs() + 1e-12)
}

fn main() {
    let sigma = 0.92;
    let z = 0.03;
    let cells = [
        ("5bp", 0.0005, 41_500.0, 2.0),
        ("5bp", 0.0005, 41_500.0, 20.0),
        ("30bp", 0.0030, 3_000.0, 2.0),
        ("30bp", 0.0030, 3_000.0, 20.0),
    ];
    let names = ["static", "gap", "defensive"];
    let n_eval = (EVAL_SEEDS.end - EVAL_SEEDS.start) as usize;
    let mut all_pass = true;

    for (sname, fee, target, arb) in cells {
        let cell = format!("{sname} s{sigma} z{z} arb{arb}");
        // hazard fixed at the 1s-benchmark calibration
        let ck_bench = &clocks()[1];
        let mut cal_cfg = config_for(fee, sigma, z, ck_bench);
        cal_cfg.arb_arrival_rate_per_hour = Some(arb);
        let (lam, achieved) = calibrate_pooled_hazard(&cal_cfg, S0, CAL_SEEDS, target, 0.05);

        // [clock][policy] means and per-seed U
        let mut mean = [[(0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64); 3]; 2]; // A,B,U,S,alloc
        let mut u_seed = vec![vec![vec![0.0f64; n_eval]; 3]; 2];
        for (ci, ck) in clocks().iter().enumerate() {
            for (si, seed) in EVAL_SEEDS.enumerate() {
                let base = gen_base(seed, sigma, lam, arb);
                let (prices, sched) = bin_base(&base, ck);
                for pi in 0..3 {
                    let mut cfg = config_for(fee, sigma, z, ck);
                    cfg.seed = seed;
                    cfg.pooled_fund_arrival_rate_per_hour = Some(lam);
                    cfg.arb_arrival_rate_per_hour = Some(arb);
                    let mut pol = make_policy(pi, fee);
                    let (records, evs) =
                        run_simulation_with_injected_schedules(&cfg, &prices, pol.as_mut(), &sched);
                    let s = summarize_events(&evs, &records);
                    let m = &mut mean[ci][pi];
                    m.0 += s.a_fill / n_eval as f64;
                    m.1 += s.b_fill / n_eval as f64;
                    m.2 += s.u_lp_rel / n_eval as f64;
                    m.3 += s.served_fund_volume / n_eval as f64;
                    m.4 += s.alloc_amm_share.unwrap_or(0.0) / n_eval as f64;
                    u_seed[ci][pi][si] = s.u_lp_rel;
                }
            }
        }
        println!("\n{cell}: lam15={lam:.1}/hr (achieved {achieved:.0}/wk)");
        for (ci, ck) in clocks().iter().enumerate() {
            println!(
                "  {:4} {}",
                ck.name,
                (0..3)
                    .map(|p| {
                        let m = mean[ci][p];
                        format!(
                            "{}: A={:.0} B={:.2} U/V0={:.3e} S={:.1} alloc={:.4}",
                            names[p],
                            m.0,
                            m.1,
                            m.2 / V0,
                            m.3,
                            m.4
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
        }
        // ── acceptance criteria ───────────────────────────────────────
        let mut cell_pass = true;
        for p in 0..2 {
            // static, gap: level criteria (5s vs the 1s benchmark)
            let (a5, b5, u5, s5, al5) = mean[0][p];
            let (a15, b15, u15, s15, al15) = mean[1][p]; // "15" = benchmark (1s)
            let checks = [
                ("dsym(A)", d_sym(a15, a5), 0.02),
                ("dsym(B)", d_sym(b15, b5), 0.02),
                ("dsym(S)", d_sym(s15, s5), 0.02),
                ("alloc_pp", (al15 - al5).abs(), 0.01),
                ("|dU|/V0", (u15 - u5).abs() / V0, 5.0e-5),
            ];
            for (label, val, thr) in checks {
                let ok = val <= thr;
                if !ok {
                    cell_pass = false;
                }
                println!(
                    "    {} {label} = {val:.5} (thr {thr}) -> {}",
                    names[p],
                    if ok { "ok" } else { "FAIL" }
                );
            }
            if u15.signum() != u5.signum() {
                cell_pass = false;
                println!("    {} U SIGN CHANGED -> FAIL", names[p]);
            }
        }
        // defensive: absolute fallbacks (5s vs 1s benchmark)
        let (a5, _, u5, s5, _) = mean[0][2];
        let (a15, _, u15, s15, _) = mean[1][2];
        let da = (a15 - a5).abs() / V0;
        let du = (u15 - u5).abs() / V0;
        let shut5 = s5 < 0.01 * mean[0][0].3;
        let shut15 = s15 < 0.01 * mean[1][0].3;
        let def_ok = da <= 1.0e-5 && du <= 1.0e-5 && shut15 == shut5;
        if !def_ok {
            cell_pass = false;
        }
        println!(
            "    defensive |dA|/V0={da:.2e} |dU|/V0={du:.2e} dS={:.2} shutdown {}=={} -> {}",
            (s15 - s5).abs(),
            shut15,
            shut5,
            if def_ok { "ok" } else { "FAIL" }
        );
        // paired policy-comparison signs
        for i in 0..3 {
            for j in (i + 1)..3 {
                let d5: f64 = (0..n_eval)
                    .map(|s| u_seed[0][i][s] - u_seed[0][j][s])
                    .sum::<f64>()
                    / n_eval as f64;
                let d15: f64 = (0..n_eval)
                    .map(|s| u_seed[1][i][s] - u_seed[1][j][s])
                    .sum::<f64>()
                    / n_eval as f64;
                let near_tie = d15.abs() / V0 < 2.0e-5; // near-tie judged on the 1s benchmark
                let agree = d5.signum() == d15.signum();
                if !agree && !near_tie {
                    cell_pass = false;
                }
                println!(
                    "    pair {}-{}: d5s/V0={:.2e} d1s/V0={:.2e} near_tie={near_tie} agree={agree}{}",
                    names[i],
                    names[j],
                    d5 / V0,
                    d15 / V0,
                    if !agree && near_tie {
                        " (disclosed, not disqualifying)"
                    } else {
                        ""
                    }
                );
            }
        }
        println!(
            "  CELL VERDICT: {}",
            if cell_pass { "PASS" } else { "FAIL" }
        );
        all_pass &= cell_pass;
    }
    println!(
        "\n== CERTIFICATION VERDICT: {} ==",
        if all_pass {
            "ALL FOUR CELLS PASS — primary clock = 5 SECONDS (frozen)"
        } else {
            "FAILED — primary clock = 1 SECOND, the finest preregistered numerical resolution (frozen; no further certification)"
        }
    );
}
