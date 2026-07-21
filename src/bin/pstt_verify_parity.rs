//! Compare two PSTT weekly JSON arrays with field-specific tolerances.
//!
//! Missing expected file is a hard failure unless `--smoke` is set.

use amm_lab::pstt::error::PsttError;
use amm_lab::pstt::parity::{FloatTol, ParityMismatch, compare_f64, compare_str, compare_u64};
use amm_lab::pstt::schema::WeeklyRow;
use clap::Parser;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(about = "PSTT weekly parity gate")]
struct Args {
    #[arg(long)]
    expected: PathBuf,
    #[arg(long)]
    actual: PathBuf,
    #[arg(long, default_value_t = false)]
    smoke: bool,
}

fn key(r: &WeeklyRow) -> String {
    format!("{}|{}|{}", r.pool, r.reference, r.week)
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(true) => {
            println!("PARITY PASS");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            println!("PARITY FAIL");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("error: {e}");
            println!("PARITY NOT CHECKED");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<bool, PsttError> {
    if !args.expected.exists() {
        if args.smoke {
            println!(
                "smoke: expected {} missing; parity NOT checked",
                args.expected.display()
            );
            return Ok(true);
        }
        return Err(PsttError::MissingPath(args.expected));
    }
    if !args.actual.exists() {
        return Err(PsttError::MissingPath(args.actual));
    }
    let expected: Vec<WeeklyRow> = serde_json::from_str(
        &fs::read_to_string(&args.expected).map_err(|e| PsttError::io(&args.expected, e))?,
    )?;
    let actual: Vec<WeeklyRow> = serde_json::from_str(
        &fs::read_to_string(&args.actual).map_err(|e| PsttError::io(&args.actual, e))?,
    )?;
    let exp: BTreeMap<_, _> = expected.iter().map(|r| (key(r), r)).collect();
    let act: BTreeMap<_, _> = actual.iter().map(|r| (key(r), r)).collect();
    let mut mismatches: Vec<ParityMismatch> = Vec::new();
    for k in exp.keys() {
        if !act.contains_key(k) {
            mismatches.push(ParityMismatch {
                key: k.clone(),
                field: "<row>".into(),
                detail: "only in expected".into(),
            });
        }
    }
    for k in act.keys() {
        if !exp.contains_key(k) {
            mismatches.push(ParityMismatch {
                key: k.clone(),
                field: "<row>".into(),
                detail: "only in actual".into(),
            });
        }
    }
    for (k, e) in &exp {
        let Some(a) = act.get(k) else { continue };
        compare_str(k, "pair", &e.pair, &a.pair, &mut mismatches);
        compare_u64(k, "fee", e.fee as u64, a.fee as u64, &mut mismatches);
        compare_f64(k, "L", e.l, a.l, FloatTol::WEEKLY, &mut mismatches);
        compare_f64(k, "A", e.a, a.a, FloatTol::WEEKLY, &mut mismatches);
        compare_f64(k, "B", e.b, a.b, FloatTol::WEEKLY, &mut mismatches);
        compare_f64(k, "S", e.s, a.s, FloatTol::WEEKLY, &mut mismatches);
        compare_u64(
            k,
            "observed_mass",
            e.observed_mass,
            a.observed_mass,
            &mut mismatches,
        );
        compare_f64(
            k,
            "service_q2",
            e.service_q2,
            a.service_q2,
            FloatTol::WEEKLY,
            &mut mismatches,
        );
        compare_u64(k, "fill_count", e.fill_count, a.fill_count, &mut mismatches);
        compare_u64(
            k,
            "matched_count",
            e.matched_count,
            a.matched_count,
            &mut mismatches,
        );
        // Identity on both sides.
        if (e.l - (e.a - e.b)).abs() > 1e-8 {
            mismatches.push(ParityMismatch {
                key: k.clone(),
                field: "identity_expected".into(),
                detail: "L != A-B".into(),
            });
        }
        if (a.l - (a.a - a.b)).abs() > 1e-8 {
            mismatches.push(ParityMismatch {
                key: k.clone(),
                field: "identity_actual".into(),
                detail: "L != A-B".into(),
            });
        }
    }
    println!("--- pstt weekly parity ---");
    println!("expected rows : {}", exp.len());
    println!("actual rows   : {}", act.len());
    println!("mismatches    : {}", mismatches.len());
    for m in mismatches.iter().take(25) {
        println!("  {} {} {}", m.key, m.field, m.detail);
    }
    Ok(mismatches.is_empty())
}
