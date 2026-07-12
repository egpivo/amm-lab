//! Panel completeness + outcome-sanity QA (pre-estimation gate).
//!
//! Reads a pool-week panel CSV (Python golden or the Rust build) and `panel_units.json`,
//! rebuilds the frozen week grid, and emits the completeness/sanity report as JSON plus a
//! terminal summary. Computes NO pre-trend and NO ATT.
//!
//! Usage:
//!   panel_report [PANEL_CSV] [PANEL_UNITS_JSON] [OUT_JSON]
//! Defaults:
//!   PANEL_CSV        = .local/amm_paper_c/data/panel_weekly_frozen.csv
//!   PANEL_UNITS_JSON = .local/amm_paper_c/data/panel_units.json
//!   OUT_JSON         = <PANEL_CSV dir>/panel_completeness_rs.json

use amm_lab::data::completeness::{frozen_week_grid, report};
use amm_lab::data::io::load_roles;
use amm_lab::data::panel::Panel;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let panel_csv = PathBuf::from(
        args.next()
            .unwrap_or_else(|| ".local/amm_paper_c/data/panel_weekly_frozen.csv".to_string()),
    );
    let units_json = PathBuf::from(
        args.next()
            .unwrap_or_else(|| ".local/amm_paper_c/data/panel_units.json".to_string()),
    );
    let out_json = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| panel_csv.with_file_name("panel_completeness_rs.json"));

    let panel = match Panel::from_csv(&panel_csv) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to read panel {}: {e}", panel_csv.display());
            return ExitCode::FAILURE;
        }
    };
    let roles = match load_roles(&units_json) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to read {}: {e}", units_json.display());
            return ExitCode::FAILURE;
        }
    };

    let grid = frozen_week_grid();
    let rep = report(&panel, &roles, &grid);

    match serde_json::to_string_pretty(&rep) {
        Ok(s) => {
            if let Err(e) = std::fs::write(&out_json, s) {
                eprintln!("warning: could not write {}: {e}", out_json.display());
            } else {
                eprintln!("wrote report -> {}", out_json.display());
            }
        }
        Err(e) => eprintln!("warning: could not serialize report: {e}"),
    }

    let obs = rep.observed_pool_weeks_total as f64;
    let exp = rep.expected_pool_weeks_total.max(1) as f64;
    println!("--- panel completeness ---");
    println!("unit set        : {}", rep.frozen_unit_set_size);
    println!("week grid       : {}", rep.frozen_week_grid_size);
    println!(
        "pool-weeks      : {} observed / {} expected ({:.1}%)",
        rep.observed_pool_weeks_total,
        rep.expected_pool_weeks_total,
        obs / exp * 100.0
    );
    println!("missing by role : {:?}", rep.missing_pool_weeks_by_role);
    println!("rows by role    : {:?}", rep.panel_rows_by_role);
    println!(
        "units w/o data  : {} (e.g. {:?})",
        rep.units_with_zero_observed_weeks_count,
        rep.units_with_zero_observed_weeks_sample
            .iter()
            .take(5)
            .collect::<Vec<_>>()
    );
    if rep.observed_weeks_outside_grid > 0 {
        println!(
            "WARN weeks outside frozen grid: {}",
            rep.observed_weeks_outside_grid
        );
    }
    println!("--- outcome sanity (min / median / max, nonzero frac) ---");
    for (name, s) in &rep.sanity {
        println!(
            "  {:<38} {:>14.4} {:>14.4} {:>16.4}   {:.3}",
            name, s.min, s.median, s.max, s.nonzero_frac
        );
    }
    ExitCode::SUCCESS
}
