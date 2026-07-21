//! Raw WETH-USDC swap tape -> oriented fills CSV (standalone M6 data path).
//!
//! Mirrors the frozen `build_m6_public.py` stage-1 fill construction:
//!   keep `type == "swap"` rows whose block has a window timestamp, whose
//!   pool is targeted, and whose `token0/token1` are canonical USDC/WETH;
//!   q = |amount1| / 1e18, p_exec = |amount0/amount1| * 1e12,
//!   direction = -1 if amount1 > 0 else +1, week = ISO %G-%V of block time.
//!
//! No network. Additive parity tool; never regenerates frozen artifacts.

use amm_lab::pstt::block_time::iso_week_label;
use amm_lab::pstt::error::PsttError;
use amm_lab::pstt::manifest::require_empty_output_dir;
use amm_lab::pstt::schema::Address;
use clap::Parser;
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(about = "PSTT standalone fill spooler: raw swap tape -> oriented fills (no network)")]
struct Args {
    /// Swap tape CSV (optionally .gz) with headers including:
    /// type,block,pool,token0,token1,amount0,amount1
    #[arg(long)]
    swaps_csv: PathBuf,
    /// Block-timestamp CSV. Positional columns (frozen layout): block at
    /// `--block-col`, unix timestamp at `--ts-col`. First row is a header.
    #[arg(long)]
    block_ts_csv: PathBuf,
    #[arg(long, default_value_t = 0)]
    block_col: usize,
    #[arg(long, default_value_t = 3)]
    ts_col: usize,
    /// JSON array of pool specs:
    /// {"pool","label","pair","fee","cex_symbol","invert","token0","token1"}
    #[arg(long)]
    pools_json: PathBuf,
    /// Inclusive lower unix bound of the block-time window.
    #[arg(long)]
    window_start_unix: i64,
    /// Exclusive upper unix bound of the block-time window.
    #[arg(long)]
    window_end_unix: i64,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_nonempty_output: bool,
}

#[derive(Debug, Deserialize)]
struct PoolSpec {
    pool: String,
    label: String,
    pair: String,
    fee: u32,
    cex_symbol: String,
    invert: bool,
    token0: String,
    token1: String,
}

fn main() -> ExitCode {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn open_maybe_gz(path: &Path) -> Result<Box<dyn Read>, PsttError> {
    let file = File::open(path).map_err(|e| PsttError::io(path, e))?;
    if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        Ok(Box::new(GzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

fn run(args: Args) -> Result<(), PsttError> {
    if args.window_end_unix <= args.window_start_unix {
        return Err(PsttError::invariant("window end must exceed window start"));
    }
    require_empty_output_dir(&args.output_dir, args.allow_nonempty_output)?;

    let pools_raw =
        fs::read_to_string(&args.pools_json).map_err(|e| PsttError::io(&args.pools_json, e))?;
    let specs: Vec<PoolSpec> = serde_json::from_str(&pools_raw)?;
    let mut targets = BTreeMap::new();
    for s in &specs {
        let addr = Address::normalize(&s.pool)?;
        let t0 = Address::normalize(&s.token0)?;
        let t1 = Address::normalize(&s.token1)?;
        if targets.insert(addr.clone(), (s, t0, t1)).is_some() {
            return Err(PsttError::DuplicateKey(format!("pool {addr}")));
        }
    }

    // Block -> timestamp, window-filtered (frozen: lo <= t < hi).
    let mut b2ts = BTreeMap::<u64, i64>::new();
    {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(open_maybe_gz(&args.block_ts_csv)?);
        for rec in rdr.records() {
            let rec = rec?;
            if rec.len() <= args.block_col.max(args.ts_col) {
                continue;
            }
            let block: u64 = match rec[args.block_col].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let ts: i64 = match rec[args.ts_col].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if args.window_start_unix <= ts && ts < args.window_end_unix {
                b2ts.insert(block, ts);
            }
        }
    }

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(open_maybe_gz(&args.swaps_csv)?);
    let headers = rdr.headers()?.clone();
    let col = |name: &str| -> Result<usize, PsttError> {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| PsttError::schema(format!("swap tape missing column {name}")))
    };
    let c_type = col("type")?;
    let c_block = col("block")?;
    let c_pool = col("pool")?;
    let c_token0 = col("token0")?;
    let c_token1 = col("token1")?;
    let c_amount0 = col("amount0")?;
    let c_amount1 = col("amount1")?;

    #[derive(Debug)]
    struct Fill {
        timestamp: i64,
        pool: Address,
        q: f64,
        p_exec: f64,
        direction: f64,
        week: String,
    }
    let mut fills: Vec<Fill> = Vec::new();
    let mut seen = 0u64;
    let mut kept = 0u64;
    for rec in rdr.records() {
        let rec = rec?;
        seen += 1;
        if &rec[c_type] != "swap" {
            continue;
        }
        let block: u64 = match rec[c_block].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some(&ts) = b2ts.get(&block) else {
            continue;
        };
        let pool = match Address::normalize(&rec[c_pool]) {
            Ok(a) => a,
            Err(_) => continue,
        };
        let Some((spec, t0, t1)) = targets.get(&pool) else {
            continue;
        };
        let a0_raw = rec[c_amount0].trim();
        let a1_raw = rec[c_amount1].trim();
        if a0_raw.is_empty() || a1_raw.is_empty() {
            continue;
        }
        // Frozen canonical-token check: token0 == USDC, token1 == WETH
        // (generalized here to the per-pool declared token pair).
        let r0 = match Address::normalize(&rec[c_token0]) {
            Ok(a) => a,
            Err(_) => continue,
        };
        let r1 = match Address::normalize(&rec[c_token1]) {
            Ok(a) => a,
            Err(_) => continue,
        };
        if r0 != *t0 || r1 != *t1 {
            continue;
        }
        let a0: i128 = a0_raw
            .parse()
            .map_err(|_| PsttError::parse(format!("amount0 not integral: {a0_raw}")))?;
        let a1: i128 = a1_raw
            .parse()
            .map_err(|_| PsttError::parse(format!("amount1 not integral: {a1_raw}")))?;
        if a1 == 0 {
            continue;
        }
        let q = (a1.unsigned_abs() as f64) / 1e18;
        let p_exec = ((a0 as f64) / (a1 as f64)).abs() * 1e12;
        let direction = if a1 > 0 { -1.0 } else { 1.0 };
        let week = iso_week_label(ts as f64)?;
        fills.push(Fill {
            timestamp: ts,
            pool: pool.clone(),
            q,
            p_exec,
            direction,
            week,
        });
        kept += 1;
        let _ = spec;
    }
    fills.sort_by(|a, b| {
        a.timestamp
            .cmp(&b.timestamp)
            .then_with(|| a.pool.cmp(&b.pool))
    });

    let fills_path = args.output_dir.join("pstt_oriented_fills.csv");
    {
        let mut wtr = csv::Writer::from_path(&fills_path)?;
        wtr.write_record([
            "timestamp",
            "pool",
            "q",
            "p_exec",
            "direction",
            "week",
            "pair",
            "fee",
            "cex_symbol",
            "invert",
        ])?;
        for f in &fills {
            let (spec, _, _) = &targets[&f.pool];
            wtr.write_record([
                format!("{}", f.timestamp),
                f.pool.to_string(),
                format!("{}", f.q),
                format!("{}", f.p_exec),
                format!("{}", f.direction),
                f.week.clone(),
                spec.pair.clone(),
                format!("{}", spec.fee),
                spec.cex_symbol.clone(),
                format!("{}", spec.invert),
            ])?;
        }
        wtr.flush().map_err(|e| PsttError::io(&fills_path, e))?;
    }

    let mut per_pool = BTreeMap::<String, u64>::new();
    for f in &fills {
        let (spec, _, _) = &targets[&f.pool];
        *per_pool.entry(spec.label.clone()).or_default() += 1;
    }
    let summary = serde_json::json!({
        "rows_seen": seen,
        "fills_kept": kept,
        "per_pool": per_pool,
        "window_unix": [args.window_start_unix, args.window_end_unix],
        "generator": "pstt_spool_fills",
        "note": "Additive Rust parity tool; not a frozen paper generator."
    });
    let summary_path = args.output_dir.join("pstt_spool_summary.json");
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)
        .map_err(|e| PsttError::io(&summary_path, e))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
