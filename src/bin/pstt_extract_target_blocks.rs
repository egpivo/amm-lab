//! Extract eligible swap-block universe. No network. Explicit paths only.
//!
//! Usage:
//!   pstt_extract_target_blocks \
//!     --events EVENTS.csv[.gz] \
//!     --pools-json POOLS.json \
//!     --start-unix START --end-unix END \
//!     --output-dir OUT

use amm_lab::pstt::error::PsttError;
use amm_lab::pstt::manifest::require_empty_output_dir;
use amm_lab::pstt::panel::{extract_eligible_blocks, union_sorted_blocks};
use amm_lab::pstt::schema::Address;
use clap::Parser;
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(about = "PSTT eligible block extraction (no network)")]
struct Args {
    #[arg(long)]
    events: PathBuf,
    /// JSON array of lowercase pool addresses, or object with `pools: [{pool: ...}]`.
    #[arg(long)]
    pools_json: PathBuf,
    #[arg(long)]
    start_unix: i64,
    #[arg(long)]
    end_unix: i64,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_nonempty_output: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PoolsInput {
    List(Vec<String>),
    Wrapped { pools: Vec<PoolAddr> },
}

#[derive(Debug, Deserialize)]
struct PoolAddr {
    pool: String,
}

fn open_events(path: &Path) -> Result<Box<dyn BufRead>, PsttError> {
    let file = File::open(path).map_err(|e| PsttError::io(path, e))?;
    if path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gz"))
    {
        Ok(Box::new(BufReader::new(GzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
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
    let raw =
        fs::read_to_string(&args.pools_json).map_err(|e| PsttError::io(&args.pools_json, e))?;
    let pools_in: PoolsInput = serde_json::from_str(&raw)?;
    let wanted: BTreeSet<String> = match pools_in {
        PoolsInput::List(v) => v
            .into_iter()
            .map(|s| Address::normalize(&s).map(|a| a.0))
            .collect::<Result<_, _>>()?,
        PoolsInput::Wrapped { pools } => pools
            .into_iter()
            .map(|p| Address::normalize(&p.pool).map(|a| a.0))
            .collect::<Result<_, _>>()?,
    };
    if wanted.is_empty() {
        return Err(PsttError::schema("pools list is empty"));
    }
    if args.end_unix <= args.start_unix {
        return Err(PsttError::schema("end_unix must be > start_unix"));
    }

    let reader = open_events(&args.events)?;
    let mut rdr = csv::Reader::from_reader(reader);
    let headers = rdr.headers()?.clone();
    let ix_pool = headers
        .iter()
        .position(|h| h == "pool")
        .ok_or_else(|| PsttError::schema("events missing pool column"))?;
    let ix_block = headers
        .iter()
        .position(|h| h == "block")
        .ok_or_else(|| PsttError::schema("events missing block column"))?;
    let ix_ts = headers
        .iter()
        .position(|h| h == "ts")
        .ok_or_else(|| PsttError::schema("events missing ts column"))?;
    let ix_type = headers
        .iter()
        .position(|h| h == "type")
        .ok_or_else(|| PsttError::schema("events missing type column"))?;

    let mut rows = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let pool = Address::normalize(rec.get(ix_pool).unwrap_or(""))?;
        let block: u64 = rec
            .get(ix_block)
            .unwrap_or("")
            .parse()
            .map_err(|_| PsttError::parse("bad block"))?;
        let ts: i64 = rec
            .get(ix_ts)
            .unwrap_or("")
            .parse()
            .map_err(|_| PsttError::parse("bad ts"))?;
        let ty = rec.get(ix_type).unwrap_or("").to_string();
        rows.push((pool, block, ts, ty));
    }

    let by_pool =
        extract_eligible_blocks(rows.into_iter(), &wanted, args.start_unix, args.end_unix)?;
    let union = union_sorted_blocks(&by_pool);

    let pool_blocks = args.output_dir.join("pstt_pool_blocks.csv");
    {
        let mut wtr = csv::Writer::from_path(&pool_blocks)?;
        wtr.write_record(["pool", "block"])?;
        for (pool, blocks) in &by_pool {
            for block in blocks {
                wtr.write_record([pool.as_str(), &block.to_string()])?;
            }
        }
        wtr.flush().map_err(|e| PsttError::io(&pool_blocks, e))?;
    }

    let targets = args.output_dir.join("pstt_target_blocks.txt");
    {
        let mut f = File::create(&targets).map_err(|e| PsttError::io(&targets, e))?;
        for b in &union {
            writeln!(f, "{b}").map_err(|e| PsttError::io(&targets, e))?;
        }
    }

    let mut counts = serde_json::Map::new();
    for (p, v) in &by_pool {
        counts.insert(p.clone(), serde_json::json!(v.len()));
    }
    let summary = serde_json::json!({
        "distinct_pools": by_pool.len(),
        "target_blocks": union.len(),
        "window_unix": [args.start_unix, args.end_unix],
        "pool_block_counts": counts,
        "generator": "pstt_extract_target_blocks",
        "note": "Additive Rust parity tool; not a frozen paper generator."
    });
    let summary_path = args.output_dir.join("pstt_block_universe.json");
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)
        .map_err(|e| PsttError::io(&summary_path, e))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
