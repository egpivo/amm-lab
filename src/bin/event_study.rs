//! Primary matched-overlap event-study harness (Eq. 3), wiring the frozen `Panel` +
//! design metadata through the causal layer to a coefficient path.
//!
//! DISCIPLINE: by default this is a *dry design check* — it assembles the matched-overlap
//! sample, validates it, and reports composition + estimability, but runs NO regression and
//! produces NO estimates. Pass `--estimate` to actually fit the two-way FE WLS and write the
//! coefficient path to an output directory (never into the paper). Even then, pre-period
//! LEAD coefficients (the pre-trend evidence) are printed first, and post-period LAG
//! coefficients are labelled PROVISIONAL: they must not be interpreted or copied into the
//! manuscript until the pre-trend has been reviewed.
//!
//! Usage:
//!   event_study [PANEL_CSV] [DATA_DIR] [--outcome NAME] [--t0 WEEK] [--estimate]
//!               [--out DIR] [--nboot N] [--alpha A] [--seed S]
//! Defaults: PANEL_CSV=.local/amm_paper_c/data/panel_weekly_frozen.csv,
//!           DATA_DIR=.local/amm_paper_c/data, outcome=twl_active_liquidity, t0=2025-51.

use amm_lab::causal::adapter::WeekGrid;
use amm_lab::causal::event_study::run;
use amm_lab::causal::{build_primary_event_study_data, load_matched_pairs, load_treatment_meta};
use amm_lab::data::completeness::frozen_week_grid;
use amm_lab::data::panel::{Panel, PoolWeek};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::ExitCode;

const OUTCOMES: &[&str] = &[
    "swaps",
    "vol0",
    "vol1",
    "twl_active_liquidity",
    "depth_1pct",
    "depth_2pct",
    "depth_5pct",
    "lp_entry_count",
    "lp_exit_count",
    "unique_lp_count",
    "jit_share_same_block",
    "lp_fee_income_native1",
    "lp_fee_income_per_active_liquidity",
    "collect_amount1_native",
    "position_duration_days",
    "net_liq",
];

fn outcome_value(r: &PoolWeek, name: &str) -> f64 {
    match name {
        "swaps" => r.swaps as f64,
        "vol0" => r.vol0,
        "vol1" => r.vol1,
        "twl_active_liquidity" => r.twl_active_liquidity,
        "depth_1pct" => r.depth_1pct,
        "depth_2pct" => r.depth_2pct,
        "depth_5pct" => r.depth_5pct,
        "lp_entry_count" => r.lp_entry_count as f64,
        "lp_exit_count" => r.lp_exit_count as f64,
        "unique_lp_count" => r.unique_lp_count as f64,
        "jit_share_same_block" => r.jit_share_same_block,
        "lp_fee_income_native1" => r.lp_fee_income_native1,
        "lp_fee_income_per_active_liquidity" => r.lp_fee_income_per_active_liquidity,
        "collect_amount1_native" => r.collect_amount1_native,
        "position_duration_days" => r.position_duration_days,
        "net_liq" => r.net_liq as f64,
        _ => f64::NAN,
    }
}

/// Outcome transform (Amendment 012). asinh handles zeros/negatives and compresses the huge
/// dynamic range so two-way FE demeaning is well-conditioned; log1p for nonneg robustness.
fn transform_value(v: f64, t: &str) -> f64 {
    match t {
        "asinh" => v.asinh(),
        "log1p" => v.max(0.0).ln_1p(),
        _ => v, // "none": raw levels (diagnostic only)
    }
}

fn opt<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let estimate = args.iter().any(|a| a == "--estimate");
    // Flags that consume the following token as their value.
    const VALUE_FLAGS: &[&str] = &[
        "--outcome",
        "--t0",
        "--out",
        "--nboot",
        "--alpha",
        "--seed",
        "--transform",
        "--horizon",
    ];
    let mut positional: Vec<&String> = Vec::new();
    let mut skip_next = false;
    for (i, a) in args.iter().enumerate() {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a.starts_with("--") {
            if VALUE_FLAGS.contains(&a.as_str()) {
                skip_next = true; // its value is not positional
            }
            continue;
        }
        let _ = i;
        positional.push(a);
    }

    let data_dir = PathBuf::from(
        positional
            .get(1)
            .map(|s| s.as_str())
            .unwrap_or(".local/amm_paper_c/data"),
    );
    let panel_csv = PathBuf::from(
        positional
            .first()
            .map(|s| s.as_str())
            .unwrap_or(".local/amm_paper_c/data/panel_weekly_frozen.csv"),
    );
    let outcome = opt(&args, "--outcome")
        .unwrap_or("twl_active_liquidity")
        .to_string();
    let t0 = opt(&args, "--t0").unwrap_or("2025-51").to_string();
    let out_dir = opt(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join("event_study_out"));
    let nboot: usize = opt(&args, "--nboot")
        .and_then(|s| s.parse().ok())
        .unwrap_or(999);
    let alpha: f64 = opt(&args, "--alpha")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.05);
    let seed: u64 = opt(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(12345);
    // Outcome transform (Amendment 012): asinh primary for liquidity magnitudes (levels FE
    // demeaning is ill-conditioned across ~20 orders of magnitude). Must match the Python spec.
    let transform = opt(&args, "--transform").unwrap_or("asinh").to_string();
    // Event-time horizon: keep |event_time| <= horizon (windowed event study).
    let horizon: Option<i64> = opt(&args, "--horizon").and_then(|s| s.parse().ok());

    if !OUTCOMES.contains(&outcome.as_str()) {
        eprintln!("unknown --outcome {outcome:?}; valid: {OUTCOMES:?}");
        return ExitCode::FAILURE;
    }

    // ---- load panel + frozen design metadata ----
    let mut panel = match Panel::from_csv(&panel_csv) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to read panel {}: {e}", panel_csv.display());
            return ExitCode::FAILURE;
        }
    };
    let meta = match load_treatment_meta(
        &data_dir.join("feerev_panelvars.csv"),
        &data_dir.join("ckpt_tokens.json"),
        &t0,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to load treatment meta: {e}");
            return ExitCode::FAILURE;
        }
    };
    let all_treated: HashSet<String> = meta
        .iter()
        .filter(|(_, m)| m.treated)
        .map(|(p, _)| p.clone())
        .collect();
    let matches = match load_matched_pairs(&data_dir.join("matched_pairs.json"), &all_treated) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to load matched pairs: {e}");
            return ExitCode::FAILURE;
        }
    };

    let grid_labels: Vec<String> = frozen_week_grid().into_iter().collect();
    let grid = match WeekGrid::new(&grid_labels) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("bad frozen week grid: {e}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!(
        "panel {} rows | {} treated | {} matched pairs | outcome={} | t0={}",
        panel.rows.len(),
        all_treated.len(),
        matches.pairs.len(),
        outcome,
        t0
    );

    // Restrict to the frozen study window: drop pool-weeks whose label is outside the grid.
    // These are boundary rows (e.g. week "2023-52" from block timestamps just before the
    // 2024-01-01 window start) -- out of scope by the pre-registered design, not the study.
    // Reconstruction still emits them (so panel<->golden parity is unaffected); the window
    // restriction lives here at the analysis layer.
    let before = panel.rows.len();
    panel.rows.retain(|r| grid.contains(&r.week));
    let dropped = before - panel.rows.len();
    if dropped > 0 {
        eprintln!("dropped {dropped} pool-weeks outside the frozen window (e.g. 2023-52 boundary)");
    }
    // Windowed event study: keep |grid.idx(week) - grid.idx(t0)| <= horizon (matches Python).
    if let Some(h) = horizon
        && let Some(t0i) = grid.idx(&t0)
    {
        let n0 = panel.rows.len();
        panel
            .rows
            .retain(|r| grid.idx(&r.week).is_some_and(|wi| (wi - t0i).abs() <= h));
        eprintln!("horizon +/-{h}: kept {} of {} rows", panel.rows.len(), n0);
    }

    // ---- assemble + validate the matched-overlap event-study design ----
    let (data, comp) = match build_primary_event_study_data(
        &panel,
        &meta,
        &grid,
        &matches,
        None, // frozen-grid completeness enforced separately by panel_report
        |r| transform_value(outcome_value(r, &outcome), &transform),
    ) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("failed to build event-study design: {e}");
            return ExitCode::FAILURE;
        }
    };

    let n_obs = data.y.len();
    let n_eff: f64 = data.weights.iter().sum();
    let mut ets: Vec<i64> = data
        .event_time
        .iter()
        .copied()
        .filter(|&k| k != -1)
        .collect();
    ets.sort_unstable();
    ets.dedup();
    println!("--- matched-overlap design ---");
    println!("matched treated units : {}", comp.matched_treated);
    println!("control units         : {}", comp.control_units);
    println!(
        "excluded unmatched trt: {}",
        comp.excluded_unmatched_treated
    );
    println!(
        "controls w/ multiplic.: {}",
        comp.control_multiplicity.len()
    );
    println!("observations (rows)   : {n_obs}");
    println!("weighted n_eff        : {n_eff:.1}");
    println!("distinct event-times  : {} (excl. -1)", ets.len());

    if !estimate {
        println!("\nDESIGN OK (dry run). No regression run, no estimates produced.");
        println!(
            "Pass --estimate to fit Eq. (3) and write the coefficient path to {}.",
            out_dir.display()
        );
        println!(
            "NOTE: estimates are for review only and must not enter the manuscript before the pre-trend is reviewed."
        );
        return ExitCode::SUCCESS;
    }

    // ---- estimate (explicit opt-in) ----
    let res = match run(&data, nboot, alpha, seed) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("estimation refused/failed: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("cannot create {}: {e}", out_dir.display());
        return ExitCode::FAILURE;
    }
    match res.write(&out_dir.to_string_lossy()) {
        Ok(()) => eprintln!("wrote coefficient path + manifest -> {}", out_dir.display()),
        Err(e) => eprintln!("warning: could not write results: {e}"),
    }

    println!(
        "\nfe_converged={} rank={} cond={:.3e} seed={} n_boot={}",
        res.fe_converged, res.rank, res.cond, res.seed, res.n_boot
    );
    println!("\n=== PRE-TREND (lead coefficients, event-time < 0; k=-1 omitted as reference) ===");
    println!(
        "  {:>4}  {:>14} {:>12} {:>12} {:>9} {:>9}",
        "k", "beta", "se", "boot_ci", "wcr_p", "wcu_p"
    );
    let mut any_lead = false;
    for (k, c) in res.bins.iter().zip(&res.coefs) {
        if *k < 0 {
            any_lead = true;
            println!(
                "  {:>4}  {:>14.4} {:>12.4} [{:.3},{:.3}] {:>9.4} {:>9.4}",
                k, c.beta, c.se, c.boot_ci_lo, c.boot_ci_hi, c.wcr_p, c.boot_p
            );
        }
    }
    if !any_lead {
        println!("  (no pre-period leads in this panel)");
    }
    println!("\n=== POST (lag coefficients, event-time >= 0) — PROVISIONAL ===");
    println!(
        "  Do NOT interpret or copy into the manuscript until the pre-trend above is reviewed."
    );
    println!(
        "  {:>4}  {:>14} {:>12} {:>12} {:>9} {:>9}",
        "k", "beta", "se", "boot_ci", "wcr_p", "wcu_p"
    );
    for (k, c) in res.bins.iter().zip(&res.coefs) {
        if *k >= 0 {
            println!(
                "  {:>4}  {:>14.4} {:>12.4} [{:.3},{:.3}] {:>9.4} {:>9.4}",
                k, c.beta, c.se, c.boot_ci_lo, c.boot_ci_hi, c.wcr_p, c.boot_p
            );
        }
    }
    ExitCode::SUCCESS
}
