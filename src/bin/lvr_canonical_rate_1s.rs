//! Round-34 P1: frozen one-second canonical LVR-rate diagnostic.
//!
//! Training seeds 20000..20099 only. No validation/final seed is read.

use amm_lab::campbell::fee_policy::FixedFeePolicy;
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{ArrivalModel, FlowRegime, SimConfig, run_simulation};
use amm_lab::campbell::summary::summarize;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

const SEED_START: u64 = 20_000;
const SEED_END_EXCLUSIVE: u64 = 20_100;
const SIGMAS: [f64; 3] = [0.48, 0.64, 0.92];
const S0: f64 = 2_000.0;
const V0: f64 = 40_000_000.0;
const HORIZON_DAYS: f64 = 7.0;
const T_YEARS: f64 = HORIZON_DAYS / 365.0;
const N_STEPS: usize = 604_800;
const DT_HOURS: f64 = 1.0 / 3_600.0;
const IDENTITY_TOL: f64 = 1e-10;
const RATIO_LOWER: f64 = 0.98;
const RATIO_UPPER: f64 = 1.02;
const ROWS_PATH: &str = ".local/lvr/canonical_rate_1s_rows.csv";
const REPORT_PATH: &str = ".local/lvr/canonical_rate_1s_report.md";

#[derive(Clone, Copy)]
struct Row {
    seed: u64,
    sigma: f64,
    a: f64,
    b: f64,
    l: f64,
    fees: f64,
    u: f64,
    target: f64,
}

impl Row {
    fn ratio(self) -> f64 {
        (self.a / V0) / self.target
    }
}

fn config(sigma: f64, seed: u64) -> SimConfig {
    SimConfig {
        name: "canonical_rate_1s".into(),
        description: "Round-34 P1 canonical arb-only diagnostic".into(),
        amm_fee: 0.0,
        cex_fee: 0.0,
        buy_demand: 0.0,
        sell_demand: 0.0,
        reserve_x: 20_000_000.0,
        reserve_y: 10_000.0,
        sigma,
        mu: 0.0,
        n_steps: N_STEPS,
        seed,
        flow_regime: FlowRegime::Normal,
        toxic_burst_prob: 0.0,
        toxic_burst_arb_scale: 1.0,
        toxic_burst_fund_scale: 1.0,
        regime_switch_period: 0,
        e1_lambda: 0.0,
        e1_fee_ref: 0.0006,
        e5_arb_prob: 1.0,
        policy_lag: 0,
        dt_hours: DT_HOURS,
        pooled_fund_arrival_rate_per_hour: None,
        buy_arrival_share: 0.5,
        arb_arrival_rate_per_hour: None,
        lookback_hours: 20.0,
        arrival_model: ArrivalModel::Bernoulli,
        log_inactive_arb: false,
    }
}

fn parse_hash_arg(name: &str) -> String {
    let args: Vec<String> = env::args().collect();
    let position = args
        .iter()
        .position(|arg| arg == name)
        .unwrap_or_else(|| panic!("missing {name}"));
    let value = args
        .get(position + 1)
        .unwrap_or_else(|| panic!("missing value for {name}"));
    assert_eq!(value.len(), 64, "{name} must be a SHA-256 digest");
    assert!(value.bytes().all(|b| b.is_ascii_hexdigit()));
    value.clone()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn standard_error(values: &[f64]) -> f64 {
    let center = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt() / (values.len() as f64).sqrt()
}

fn identity_scale(row: Row) -> f64 {
    row.a.abs().max(row.l.abs()).max(row.u.abs()).max(1.0)
}

fn identity_ok(row: Row) -> bool {
    let scale = identity_scale(row);
    row.b.abs() / scale <= IDENTITY_TOL
        && (row.a - row.l).abs() / scale <= IDENTITY_TOL
        && row.fees.abs() / scale <= IDENTITY_TOL
        && (row.u + row.l).abs() / scale <= IDENTITY_TOL
}

fn main() {
    let plan_sha256 = parse_hash_arg("--plan-sha256");
    let runner_sha256 = parse_hash_arg("--runner-sha256");
    let mut rows = Vec::with_capacity(SIGMAS.len() * (SEED_END_EXCLUSIVE - SEED_START) as usize);

    for sigma in SIGMAS {
        let target = sigma.powi(2) * T_YEARS / 8.0;
        for seed in SEED_START..SEED_END_EXCLUSIVE {
            let prices = generate_gbm(N_STEPS, S0, 0.0, sigma, T_YEARS / N_STEPS as f64, seed);
            let cfg = config(sigma, seed);
            let mut policy = FixedFeePolicy::new(0.0);
            let records = run_simulation(&cfg, &prices, &mut policy);
            let summary = summarize(&records);
            let row = Row {
                seed,
                sigma,
                a: summary.a_fill,
                b: summary.b_fill,
                l: summary.l_total,
                fees: summary.fees_total,
                u: summary.u_lp_rel,
                target,
            };
            assert!(
                identity_ok(row),
                "identity failure at sigma={sigma}, seed={seed}"
            );
            rows.push(row);
        }
        eprintln!("completed sigma={sigma}");
    }

    let mut csv = BufWriter::new(File::create(ROWS_PATH).expect("create rows CSV"));
    writeln!(
        csv,
        "seed,sigma,horizon_days,n_steps,dt_seconds,a,b,l,fees,u,a_over_v0,b_over_v0,l_over_v0,u_over_v0,canonical_target,ratio_to_target"
    )
    .unwrap();
    for row in &rows {
        writeln!(
            csv,
            "{},{:.2},{:.0},{},{:.0},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            row.seed,
            row.sigma,
            HORIZON_DAYS,
            N_STEPS,
            DT_HOURS * 3600.0,
            row.a,
            row.b,
            row.l,
            row.fees,
            row.u,
            row.a / V0,
            row.b / V0,
            row.l / V0,
            row.u / V0,
            row.target,
            row.ratio(),
        )
        .unwrap();
    }
    csv.flush().unwrap();

    let mut report = BufWriter::new(File::create(REPORT_PATH).expect("create report"));
    writeln!(report, "# One-Second Canonical-Rate Diagnostic\n").unwrap();
    writeln!(report, "## Frozen configuration\n").unwrap();
    writeln!(report, "- Training seeds: `20000..20099` (100 per sigma).").unwrap();
    writeln!(report, "- Annualized sigma: `{{0.48,0.64,0.92}}`.").unwrap();
    writeln!(report, "- Horizon: one week; 604,800 one-second steps.").unwrap();
    writeln!(
        report,
        "- Arbitrage offered every step; zero AMM and outside fees."
    )
    .unwrap();
    writeln!(
        report,
        "- No fundamental arrivals; fixed constant-product liquidity."
    )
    .unwrap();
    writeln!(
        report,
        "- Driftless validation-grid martingale convention (`mu=0` GBM).\n"
    )
    .unwrap();
    writeln!(report, "## Results\n").unwrap();
    writeln!(report, "| sigma | mean A/V0 | MC SE | mean L/V0 | mean B/V0 | target | ratio | ratio SE | error | gate |").unwrap();
    writeln!(
        report,
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"
    )
    .unwrap();
    let mut all_pass = true;
    for sigma in SIGMAS {
        let group: Vec<Row> = rows
            .iter()
            .copied()
            .filter(|row| row.sigma == sigma)
            .collect();
        let a_v0: Vec<f64> = group.iter().map(|row| row.a / V0).collect();
        let l_v0: Vec<f64> = group.iter().map(|row| row.l / V0).collect();
        let b_v0: Vec<f64> = group.iter().map(|row| row.b / V0).collect();
        let ratios: Vec<f64> = group.iter().map(|row| row.ratio()).collect();
        let target = sigma.powi(2) * T_YEARS / 8.0;
        let ratio = mean(&ratios);
        let pass = (RATIO_LOWER..=RATIO_UPPER).contains(&ratio);
        all_pass &= pass;
        writeln!(
            report,
            "| {sigma:.2} | {:.9} | {:.2e} | {:.9} | {:.2e} | {:.9} | {:.6} | {:.2e} | {:+.3}% | {} |",
            mean(&a_v0),
            standard_error(&a_v0),
            mean(&l_v0),
            mean(&b_v0),
            target,
            ratio,
            standard_error(&ratios),
            (ratio - 1.0) * 100.0,
            if pass { "PASS" } else { "FAIL" },
        )
        .unwrap();
    }
    let max_b = rows.iter().map(|row| row.b.abs() / V0).fold(0.0, f64::max);
    let max_a_l = rows
        .iter()
        .map(|row| (row.a - row.l).abs() / V0)
        .fold(0.0, f64::max);
    let max_fees = rows
        .iter()
        .map(|row| row.fees.abs() / V0)
        .fold(0.0, f64::max);
    let max_u_l = rows
        .iter()
        .map(|row| (row.u + row.l).abs() / V0)
        .fold(0.0, f64::max);
    writeln!(report, "\n## Identity checks\n").unwrap();
    writeln!(
        report,
        "All 300 rows satisfy the pre-specified numeric tolerance `{IDENTITY_TOL}`."
    )
    .unwrap();
    writeln!(report, "- max `abs(B)/V0`: `{max_b:.3e}`.").unwrap();
    writeln!(report, "- max `abs(A-L)/V0`: `{max_a_l:.3e}`.").unwrap();
    writeln!(report, "- max `abs(fees)/V0`: `{max_fees:.3e}`.").unwrap();
    writeln!(report, "- max `abs(U+L)/V0`: `{max_u_l:.3e}`.\n").unwrap();
    writeln!(report, "## Coarser-clock context\n").unwrap();
    writeln!(report, "The earlier 30-day, sigma=0.4 smoke diagnostic reported ratios 0.9934 at one hour, 0.9933 at 15 minutes, and 0.9972 at 3.75 minutes. Those settings are not like-for-like with this one-week sigma grid, but they show the expected movement toward one as the clock is refined; the present one-second gate is the frozen endpoint check.\n").unwrap();
    writeln!(report, "## Verdict\n").unwrap();
    writeln!(
        report,
        "**{}**: every sigma ratio {} `[0.98,1.02]`; P2--P4 {}.\n",
        if all_pass { "PASS" } else { "FAIL" },
        if all_pass {
            "lies in"
        } else {
            "does not lie in"
        },
        if all_pass {
            "remain blocked pending reviewer approval"
        } else {
            "must not run"
        },
    )
    .unwrap();
    writeln!(report, "## Integrity\n").unwrap();
    writeln!(report, "- Plan SHA-256: `{plan_sha256}`.").unwrap();
    writeln!(report, "- Runner SHA-256: `{runner_sha256}`.").unwrap();
    writeln!(report, "- Output rows: 300.").unwrap();
    writeln!(report, "- Validation/final seeds read: none.").unwrap();
    writeln!(report, "- Calibration or policy changes: none.").unwrap();
    report.flush().unwrap();

    println!("wrote {ROWS_PATH}");
    println!("wrote {REPORT_PATH}");
    println!("verdict={}", if all_pass { "PASS" } else { "FAIL" });
}
