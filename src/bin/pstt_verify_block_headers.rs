//! Deterministic offline verification of a fetched block-header CSV.
//!
//! Gates:
//!  1. full coverage of the requested block list, no duplicates;
//!  2. timestamps nondecreasing in block order;
//!  3. sparse parent-chain consistency: whenever blocks n and n+1 are both
//!     present, `parent_hash(n+1) == block_hash(n)`;
//!  4. well-formed 32-byte hex hash fields.
//!
//! No network. Additive parity tool; never regenerates frozen artifacts.

use amm_lab::pstt::block_time::{sparse_parent_chain_ok, timestamps_nondecreasing};
use amm_lab::pstt::error::PsttError;
use clap::Parser;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(about = "PSTT block-header deterministic verifier (no network)")]
struct Args {
    /// CSV: block,block_hash,parent_hash,timestamp_unix (with header row).
    #[arg(long)]
    headers_csv: PathBuf,
    /// Optional newline-separated required block list for coverage.
    #[arg(long)]
    blocks_file: Option<PathBuf>,
    /// Optional JSON report output path.
    #[arg(long)]
    report_json: Option<PathBuf>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn is_hash(s: &str) -> bool {
    s.len() == 66 && s.starts_with("0x") && s[2..].chars().all(|c| c.is_ascii_hexdigit())
}

fn run(args: Args) -> Result<bool, PsttError> {
    let f = fs::File::open(&args.headers_csv).map_err(|e| PsttError::io(&args.headers_csv, e))?;
    let mut rows: BTreeMap<u64, (String, String, i64)> = BTreeMap::new();
    let mut duplicates = Vec::new();
    let mut malformed = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line.map_err(|e| PsttError::io(&args.headers_csv, e))?;
        if i == 0 && line.starts_with("block,") {
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 4 {
            malformed.push(i + 1);
            continue;
        }
        let block: u64 = match parts[0].trim().parse() {
            Ok(v) => v,
            Err(_) => {
                malformed.push(i + 1);
                continue;
            }
        };
        let hash = parts[1].trim().to_lowercase();
        let parent = parts[2].trim().to_lowercase();
        let ts: i64 = match parts[3].trim().parse() {
            Ok(v) => v,
            Err(_) => {
                malformed.push(i + 1);
                continue;
            }
        };
        if !is_hash(&hash) || !is_hash(&parent) {
            malformed.push(i + 1);
            continue;
        }
        if rows.insert(block, (hash, parent, ts)).is_some() {
            duplicates.push(block);
        }
    }

    let mut missing = Vec::new();
    if let Some(blocks_file) = &args.blocks_file {
        let raw = fs::read_to_string(blocks_file).map_err(|e| PsttError::io(blocks_file, e))?;
        for l in raw.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let b: u64 = l
                .parse()
                .map_err(|_| PsttError::parse(format!("bad block number: {l}")))?;
            if !rows.contains_key(&b) {
                missing.push(b);
            }
        }
    }

    let ordered: Vec<(u64, i64)> = rows.iter().map(|(b, (_, _, t))| (*b, *t)).collect();
    let ts_ok = timestamps_nondecreasing(&ordered);
    let hdrs: Vec<(u64, &str, &str)> = rows
        .iter()
        .map(|(b, (h, p, _))| (*b, h.as_str(), p.as_str()))
        .collect();
    let chain_ok = sparse_parent_chain_ok(&hdrs);

    let pass = duplicates.is_empty()
        && malformed.is_empty()
        && missing.is_empty()
        && ts_ok
        && chain_ok
        && !rows.is_empty();
    let report = serde_json::json!({
        "headers": rows.len(),
        "duplicates": duplicates,
        "malformed_lines": malformed,
        "missing_blocks": missing.len(),
        "missing_sample": missing.iter().take(10).collect::<Vec<_>>(),
        "timestamps_nondecreasing": ts_ok,
        "sparse_parent_chain_ok": chain_ok,
        "pass": pass,
        "generator": "pstt_verify_block_headers",
    });
    if let Some(path) = &args.report_json {
        fs::write(path, serde_json::to_vec_pretty(&report)?).map_err(|e| PsttError::io(path, e))?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(pass)
}
