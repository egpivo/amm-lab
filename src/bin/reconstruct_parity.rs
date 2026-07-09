//! Golden-parity acceptance gate for the Rust reconstruction port.
//!
//! Loads a build_outcomes-style data directory, runs the Rust `reconstruct`, and compares
//! the result row-for-row against the Python golden panel (`panel_weekly_frozen.csv`) via
//! `Panel::compare`. Integer fields must match exactly; floats within `Tol`.
//!
//! Usage:
//!   reconstruct_parity [DATA_DIR] [GOLDEN_CSV] [--smoke]
//! Defaults: DATA_DIR=.local/amm_paper_c/data,
//!           GOLDEN_CSV=DATA_DIR/panel_weekly_frozen.csv (where build_outcomes.py writes it).
//!
//! Exit code 0 iff parity passes. A MISSING golden is a hard failure (exit 1, "PARITY NOT
//! CHECKED") so automation can never mistake "not compared" for "parity passed"; pass
//! `--smoke` (alias `--allow-missing-golden`) to run reconstruction only and exit 0. The
//! Rust panel is always written to DATA_DIR/panel_weekly_rust.csv for inspection.

use amm_lab::data::panel::{Panel, Tol, compare};
use amm_lab::data::reconstruct_dir_streaming;
use std::path::PathBuf;
use std::process::ExitCode;

/// Number of hash-partition shards; peak memory is ~1/N of the events file.
const N_SHARDS: usize = 64;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `--smoke` (alias `--allow-missing-golden`) turns a missing golden into a successful
    // reconstruction-only run; without it a missing golden is a HARD FAILURE so that
    // automation can never mistake "not compared" for "parity passed".
    let smoke = args
        .iter()
        .any(|a| a == "--smoke" || a == "--allow-missing-golden");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();

    let data_dir = PathBuf::from(
        positional
            .first()
            .map(|s| s.as_str())
            .unwrap_or(".local/amm_paper_c/data"),
    );
    // build_outcomes.py writes the golden to DATA/panel_weekly_frozen.csv (not DATA/processed).
    let golden = positional
        .get(1)
        .map(|s| PathBuf::from(s.as_str()))
        .unwrap_or_else(|| data_dir.join("panel_weekly_frozen.csv"));

    eprintln!(
        "streaming reconstruction from {} ({N_SHARDS} shards)",
        data_dir.display()
    );
    let rust = match reconstruct_dir_streaming(&data_dir, N_SHARDS) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to reconstruct: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("reconstructed {} pool-week rows", rust.rows.len());

    // Write next to the data dir (always exists), not next to the golden (whose parent may not).
    let rust_out = data_dir.join("panel_weekly_rust.csv");
    if let Err(e) = rust.to_csv(&rust_out) {
        eprintln!("warning: could not write {}: {e}", rust_out.display());
    } else {
        eprintln!("wrote Rust panel -> {}", rust_out.display());
    }

    if !golden.exists() {
        let pools: std::collections::HashSet<&str> =
            rust.rows.iter().map(|r| r.pool.as_str()).collect();
        if smoke {
            println!("--- smoke run (--smoke, no golden) ---");
            println!(
                "golden {} not found; wrote Rust panel only, parity NOT checked",
                golden.display()
            );
            println!(
                "rust pool-week rows: {} across {} pools",
                rust.rows.len(),
                pools.len()
            );
            return ExitCode::SUCCESS;
        }
        eprintln!(
            "ERROR: golden {} not found. Parity was NOT checked. \
             Pass --smoke to run reconstruction only (exit 0), otherwise this is a failure.",
            golden.display()
        );
        println!("PARITY NOT CHECKED (golden missing)");
        return ExitCode::FAILURE;
    }

    eprintln!("loading golden {}", golden.display());
    let gold = match Panel::from_csv(&golden) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to read golden panel: {e}");
            return ExitCode::FAILURE;
        }
    };

    let rep = compare(&gold, &rust, Tol::default());
    println!("--- parity report ---");
    println!("golden keys : {}", rep.keys_a);
    println!("rust keys   : {}", rep.keys_b);
    println!("common keys : {}", rep.common_keys);
    println!("only in gold: {}", rep.only_in_a.len());
    println!("only in rust: {}", rep.only_in_b.len());
    println!("mismatches  : {}", rep.mismatches.len());

    for (p, w) in rep.only_in_a.iter().take(10) {
        println!("  only-gold  {p} {w}");
    }
    for (p, w) in rep.only_in_b.iter().take(10) {
        println!("  only-rust  {p} {w}");
    }
    for m in rep.mismatches.iter().take(20) {
        println!(
            "  mismatch {} {} {}: gold={} rust={} (|d|={:.3e})",
            m.pool,
            m.week,
            m.field,
            m.a,
            m.b,
            (m.a - m.b).abs()
        );
    }

    if rep.is_pass() {
        println!("PARITY PASS");
        ExitCode::SUCCESS
    } else {
        println!("PARITY FAIL");
        ExitCode::FAILURE
    }
}
