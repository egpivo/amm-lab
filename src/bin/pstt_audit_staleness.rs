//! Strict-pre-block last-trade staleness audit over CSV trade tapes (no network, no zip).
//!
//! Trade files are headerless Binance aggTrades CSVs named `{SYMBOL}-aggTrades-{YYYY-MM-DD}.csv`
//! under `--trades-dir/{SYMBOL}/`.

use amm_lab::pstt::asof::strict_pre_timestamp;
use amm_lab::pstt::block_time::utc_day;
use amm_lab::pstt::cex::load_aggtrades_csv;
use amm_lab::pstt::diagnostics::{StalenessGate, gate_pass, pool_staleness_stats};
use amm_lab::pstt::error::PsttError;
use amm_lab::pstt::manifest::require_empty_output_dir;
use amm_lab::pstt::schema::{AggTrade, ContrastRecord};
use chrono::{Duration, NaiveDate};
use clap::Parser;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(about = "PSTT staleness audit (CSV references, no network)")]
struct Args {
    #[arg(long)]
    inventory_json: PathBuf,
    #[arg(long)]
    block_timestamps_csv: PathBuf,
    #[arg(long)]
    pool_blocks_csv: PathBuf,
    #[arg(long)]
    trades_dir: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value = "2024-01-01")]
    start_day: String,
    #[arg(long, default_value = "2025-12-27")]
    end_day: String,
    #[arg(long, default_value_t = false)]
    allow_nonempty_output: bool,
}

#[derive(Debug, Deserialize)]
struct Inventory {
    contrasts: Vec<ContrastRecord>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(args: Args) -> Result<(), PsttError> {
    require_empty_output_dir(&args.output_dir, args.allow_nonempty_output)?;
    let inv: Inventory = serde_json::from_str(
        &fs::read_to_string(&args.inventory_json)
            .map_err(|e| PsttError::io(&args.inventory_json, e))?,
    )?;
    let mut bts: BTreeMap<u64, i64> = BTreeMap::new();
    {
        let mut rdr = csv::Reader::from_path(&args.block_timestamps_csv)?;
        for row in rdr.deserialize::<BTreeMap<String, String>>() {
            let row = row?;
            let block: u64 = row
                .get("block_number")
                .ok_or_else(|| PsttError::schema("missing block_number"))?
                .parse()
                .map_err(|_| PsttError::parse("bad block_number"))?;
            let ts: i64 = row
                .get("timestamp_unix")
                .ok_or_else(|| PsttError::schema("missing timestamp_unix"))?
                .parse()
                .map_err(|_| PsttError::parse("bad timestamp_unix"))?;
            bts.insert(block, ts);
        }
    }
    let mut pool_blocks: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    {
        let mut rdr = csv::Reader::from_path(&args.pool_blocks_csv)?;
        for row in rdr.deserialize::<BTreeMap<String, String>>() {
            let row = row?;
            let pool = row
                .get("pool")
                .ok_or_else(|| PsttError::schema("missing pool"))?
                .to_ascii_lowercase();
            let block: u64 = row
                .get("block")
                .ok_or_else(|| PsttError::schema("missing block"))?
                .parse()
                .map_err(|_| PsttError::parse("bad block"))?;
            pool_blocks.entry(pool).or_default().push(block);
        }
    }

    let mut by_symbol: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for c in &inv.contrasts {
        by_symbol
            .entry(c.cex_symbol.clone())
            .or_default()
            .insert(c.pool_lower.as_str().to_string());
        by_symbol
            .entry(c.cex_symbol.clone())
            .or_default()
            .insert(c.pool_higher.as_str().to_string());
    }

    let start = NaiveDate::parse_from_str(&args.start_day, "%Y-%m-%d")
        .map_err(|e| PsttError::parse(e.to_string()))?;
    let end = NaiveDate::parse_from_str(&args.end_day, "%Y-%m-%d")
        .map_err(|e| PsttError::parse(e.to_string()))?;

    let mut pool_stats = BTreeMap::new();
    let invert_by_symbol: BTreeMap<String, bool> = inv
        .contrasts
        .iter()
        .map(|c| (c.cex_symbol.clone(), c.invert))
        .collect();

    for (symbol, pools) in by_symbol {
        let invert = *invert_by_symbol.get(&symbol).unwrap_or(&false);
        let mut blocks_by_day: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
        for pool in &pools {
            for &b in pool_blocks.get(pool).into_iter().flatten() {
                let ts = *bts
                    .get(&b)
                    .ok_or_else(|| PsttError::MissingJoin(format!("missing block ts {b}")))?;
                let day = utc_day(ts as f64)?.to_string();
                blocks_by_day
                    .entry((pool.clone(), day))
                    .or_default()
                    .push(ts as f64);
            }
        }
        let mut stale: BTreeMap<String, Vec<f64>> =
            pools.iter().map(|p| (p.clone(), Vec::new())).collect();
        let mut joined: BTreeMap<String, u64> = pools.iter().map(|p| (p.clone(), 0u64)).collect();
        let mut prior: Option<f64> = None;
        let mut day = start;
        while day <= end {
            let ds = day.to_string();
            let path = args
                .trades_dir
                .join(&symbol)
                .join(format!("{symbol}-aggTrades-{ds}.csv"));
            let times: Vec<AggTrade> = if path.exists() {
                load_aggtrades_csv(&path, invert)?
            } else {
                Vec::new()
            };
            let trade_ts: Vec<AggTrade> = times.clone();
            for pool in &pools {
                for &t in blocks_by_day
                    .get(&(pool.clone(), ds.clone()))
                    .into_iter()
                    .flatten()
                {
                    match strict_pre_timestamp(&trade_ts, t) {
                        Ok(j) => {
                            *joined.get_mut(pool).unwrap() += 1;
                            stale.get_mut(pool).unwrap().push(j.staleness_seconds);
                        }
                        Err(_) => {
                            if let Some(p) = prior {
                                *joined.get_mut(pool).unwrap() += 1;
                                stale.get_mut(pool).unwrap().push(t - p);
                            }
                        }
                    }
                }
            }
            if let Some(last) = trade_ts.last() {
                prior = Some(last.timestamp_secs);
            }
            day += Duration::days(1);
        }
        for pool in pools {
            let total = pool_blocks.get(&pool).map(|v| v.len() as u64).unwrap_or(0);
            let mut values = stale.remove(&pool).unwrap_or_default();
            pool_stats.insert(pool, pool_staleness_stats(total, &mut values));
        }
    }

    let gate = StalenessGate::default();
    let mut surviving = Vec::new();
    let mut limited = Vec::new();
    let mut contrast_rows = Vec::new();
    for c in &inv.contrasts {
        let lo = pool_stats
            .get(c.pool_lower.as_str())
            .ok_or_else(|| PsttError::schema(format!("missing stats {}", c.pool_lower)))?;
        let hi = pool_stats
            .get(c.pool_higher.as_str())
            .ok_or_else(|| PsttError::schema(format!("missing stats {}", c.pool_higher)))?;
        let passed = gate_pass(lo, gate) && gate_pass(hi, gate);
        if passed {
            surviving.push(c.contrast_id.clone());
        } else {
            limited.push(c.contrast_id.clone());
        }
        contrast_rows.push(serde_json::json!({
            "contrast_id": c.contrast_id,
            "pair": c.pair,
            "pool_lower": c.pool_lower,
            "pool_higher": c.pool_higher,
            "pool_lower_staleness": lo,
            "pool_higher_staleness": hi,
            "gate_pass": passed,
            "status": if passed { "SURVIVES" } else { "REFERENCE-LIMITED" },
        }));
    }

    let audit = serde_json::json!({
        "gate": {"q99_seconds_max": gate.q99_seconds_max, "coverage_min": gate.coverage_min},
        "pool_stats": pool_stats,
        "contrasts": contrast_rows,
        "surviving": surviving,
        "reference_limited": limited,
        "generator": "pstt_audit_staleness",
        "note": "Additive Rust parity tool; not a frozen paper generator."
    });
    let audit_path = args.output_dir.join("pstt_staleness_audit.json");
    fs::write(&audit_path, serde_json::to_vec_pretty(&audit)?)
        .map_err(|e| PsttError::io(&audit_path, e))?;

    let inclusion = serde_json::json!({
        "spec": "pstt_parity",
        "surviving_contrasts": inv.contrasts.iter().filter(|c| surviving.contains(&c.contrast_id)).collect::<Vec<_>>(),
        "reference_limited": limited,
        "generator": "pstt_audit_staleness"
    });
    let inclusion_path = args.output_dir.join("pstt_inclusion.json");
    fs::write(&inclusion_path, serde_json::to_vec_pretty(&inclusion)?)
        .map_err(|e| PsttError::io(&inclusion_path, e))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "surviving": surviving,
            "reference_limited": limited
        }))?
    );
    Ok(())
}
