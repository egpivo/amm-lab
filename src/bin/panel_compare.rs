//! Compare two pool-week panels (row-for-row by (pool, week)) via `Panel::compare`.
//!
//! Generic parity check usable for:
//!   - full golden vs Rust  (expect only_in_a = only_in_b = mismatches = 0)
//!   - a SUBSET golden vs the full Rust panel: `only_in_b` will list every pool absent from
//!     the subset (expected), so `--subset` judges the pass on the golden's scope only, i.e.
//!     `only_in_a == 0 && mismatches == 0` (every golden row reproduced by Rust, all agree).
//!
//! Usage: panel_compare GOLDEN_CSV RUST_CSV [--subset]

use amm_lab::data::panel::{Panel, Tol, compare};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let subset = args.iter().any(|a| a == "--subset");
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if pos.len() < 2 {
        eprintln!("usage: panel_compare GOLDEN_CSV RUST_CSV [--subset]");
        return ExitCode::FAILURE;
    }
    let golden_path = PathBuf::from(pos[0]);
    let rust_path = PathBuf::from(pos[1]);

    let golden = match Panel::from_csv(&golden_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to read golden {}: {e}", golden_path.display());
            return ExitCode::FAILURE;
        }
    };
    let rust = match Panel::from_csv(&rust_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to read rust {}: {e}", rust_path.display());
            return ExitCode::FAILURE;
        }
    };

    let rep = compare(&golden, &rust, Tol::default());
    println!(
        "--- panel compare (a=golden {}, b=rust {}) ---",
        golden_path.display(),
        rust_path.display()
    );
    println!("golden keys      : {}", rep.keys_a);
    println!("rust keys        : {}", rep.keys_b);
    println!("common keys      : {}", rep.common_keys);
    println!("only in golden   : {}", rep.only_in_a.len());
    println!("only in rust     : {}", rep.only_in_b.len());
    println!("field mismatches : {}", rep.mismatches.len());
    for (p, w) in rep.only_in_a.iter().take(10) {
        println!("  only-golden  {p} {w}");
    }
    for m in rep.mismatches.iter().take(25) {
        println!(
            "  mismatch {} {} {}: golden={} rust={} (|d|={:.3e})",
            m.pool,
            m.week,
            m.field,
            m.a,
            m.b,
            (m.a - m.b).abs()
        );
    }

    // Pass criteria: full mode requires exact key sets + no mismatch; subset mode judges only
    // the golden's scope (every golden row reproduced by Rust and agreeing).
    let pass = if subset {
        rep.only_in_a.is_empty() && rep.mismatches.is_empty()
    } else {
        rep.is_pass()
    };
    if subset {
        println!(
            "(subset mode: only_in_rust={} ignored)",
            rep.only_in_b.len()
        );
    }
    if pass {
        println!("PARITY PASS");
        ExitCode::SUCCESS
    } else {
        println!("PARITY FAIL");
        ExitCode::FAILURE
    }
}
