//! Ethereum block-header fetch (JSON-RPC `eth_getBlockByNumber`) with
//! resumable checkpointing. The ONLY networked PSTT tool; every other stage
//! is strictly offline. Deterministic verification lives in the separate
//! offline `pstt_verify_block_headers` binary.
//!
//! Output: append-only CSV `block,block_hash,parent_hash,timestamp_unix`.
//! Additive parity tool; never regenerates frozen artifacts.

use amm_lab::pstt::error::PsttError;
use clap::Parser;
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(about = "PSTT Ethereum block-header fetcher (JSON-RPC, resumable)")]
struct Args {
    /// JSON-RPC endpoint URL (http/https).
    #[arg(long)]
    rpc_url: String,
    /// Newline-separated block numbers to fetch.
    #[arg(long)]
    blocks_file: PathBuf,
    /// Output CSV (created with header if absent; appended if present).
    #[arg(long)]
    output_csv: PathBuf,
    /// Blocks per JSON-RPC batch request.
    #[arg(long, default_value_t = 100)]
    batch_size: usize,
    /// Milliseconds to sleep between batches.
    #[arg(long, default_value_t = 200)]
    sleep_ms: u64,
    /// Stop after this many blocks (0 = no cap); useful for smoke runs.
    #[arg(long, default_value_t = 0)]
    max_blocks: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn parse_hex_u64(s: &str) -> Result<u64, PsttError> {
    let t = s.trim_start_matches("0x");
    u64::from_str_radix(t, 16).map_err(|_| PsttError::parse(format!("bad hex quantity: {s}")))
}

fn run(args: Args) -> Result<(), PsttError> {
    let wanted: Vec<u64> = {
        let raw = fs::read_to_string(&args.blocks_file)
            .map_err(|e| PsttError::io(&args.blocks_file, e))?;
        let mut v: Vec<u64> = raw
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(|l| {
                l.parse::<u64>()
                    .map_err(|_| PsttError::parse(format!("bad block number: {l}")))
            })
            .collect::<Result<_, _>>()?;
        v.sort_unstable();
        v.dedup();
        v
    };

    // Resume: skip blocks already checkpointed.
    let mut have = BTreeSet::new();
    if args.output_csv.exists() {
        let f = fs::File::open(&args.output_csv).map_err(|e| PsttError::io(&args.output_csv, e))?;
        for (i, line) in BufReader::new(f).lines().enumerate() {
            let line = line.map_err(|e| PsttError::io(&args.output_csv, e))?;
            if i == 0 && line.starts_with("block,") {
                continue;
            }
            if let Some(Ok(b)) = line.split(',').next().map(|s| s.trim().parse::<u64>()) {
                have.insert(b);
            }
        }
    } else {
        let mut f =
            fs::File::create(&args.output_csv).map_err(|e| PsttError::io(&args.output_csv, e))?;
        writeln!(f, "block,block_hash,parent_hash,timestamp_unix")
            .map_err(|e| PsttError::io(&args.output_csv, e))?;
    }
    let todo: Vec<u64> = wanted
        .iter()
        .copied()
        .filter(|b| !have.contains(b))
        .collect();
    let todo = if args.max_blocks > 0 && todo.len() > args.max_blocks {
        todo[..args.max_blocks].to_vec()
    } else {
        todo
    };
    eprintln!(
        "wanted={} already={} fetching={}",
        wanted.len(),
        have.len(),
        todo.len()
    );

    let mut out = OpenOptions::new()
        .append(true)
        .open(&args.output_csv)
        .map_err(|e| PsttError::io(&args.output_csv, e))?;
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(30))
        .build();
    let mut fetched = 0usize;
    for chunk in todo.chunks(args.batch_size.max(1)) {
        let batch: Vec<serde_json::Value> = chunk
            .iter()
            .enumerate()
            .map(|(i, b)| {
                serde_json::json!({
                    "jsonrpc": "2.0", "id": i,
                    "method": "eth_getBlockByNumber",
                    "params": [format!("0x{b:x}"), false],
                })
            })
            .collect();
        let resp = agent
            .post(&args.rpc_url)
            .set("content-type", "application/json")
            .send_string(&serde_json::to_string(&batch)?)
            .map_err(|e| PsttError::parse(format!("rpc error: {e}")))?;
        let body: serde_json::Value = resp
            .into_json()
            .map_err(|e| PsttError::parse(format!("rpc response not json: {e}")))?;
        let items = body
            .as_array()
            .ok_or_else(|| PsttError::schema("batched rpc response must be an array"))?;
        for item in items {
            let id = item["id"].as_u64().unwrap_or(u64::MAX) as usize;
            let block_number = *chunk.get(id).ok_or_else(|| {
                PsttError::schema(format!("rpc response id {id} out of batch range"))
            })?;
            let result = &item["result"];
            if result.is_null() {
                return Err(PsttError::MissingJoin(format!(
                    "rpc returned null header for block {block_number}"
                )));
            }
            let number = parse_hex_u64(result["number"].as_str().unwrap_or_default())?;
            if number != block_number {
                return Err(PsttError::invariant(format!(
                    "rpc returned block {number}, requested {block_number}"
                )));
            }
            let hash = result["hash"].as_str().unwrap_or_default().to_lowercase();
            let parent = result["parentHash"]
                .as_str()
                .unwrap_or_default()
                .to_lowercase();
            let ts = parse_hex_u64(result["timestamp"].as_str().unwrap_or_default())? as i64;
            if hash.len() != 66 || parent.len() != 66 {
                return Err(PsttError::schema(format!(
                    "malformed hash fields for block {block_number}"
                )));
            }
            writeln!(out, "{number},{hash},{parent},{ts}")
                .map_err(|e| PsttError::io(&args.output_csv, e))?;
            fetched += 1;
        }
        out.flush()
            .map_err(|e| PsttError::io(&args.output_csv, e))?;
        eprintln!("checkpoint: +{} (total fetched {fetched})", chunk.len());
        std::thread::sleep(std::time::Duration::from_millis(args.sleep_ms));
    }
    println!(
        "{}",
        serde_json::json!({
            "fetched": fetched,
            "output": args.output_csv,
            "generator": "pstt_fetch_block_headers",
            "note": "Run pstt_verify_block_headers before using this file."
        })
    );
    Ok(())
}
