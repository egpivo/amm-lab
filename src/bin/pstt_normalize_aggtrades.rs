//! Binance aggTrades archive normalization (standalone public-application data path).
//!
//! Reads daily archives named `{SYMBOL}-aggTrades-YYYY-MM-DD.{zip|csv|csv.gz}`
//! from an input directory and writes normalized headerless CSVs
//! `{SYMBOL}/{SYMBOL}-aggTrades-YYYY-MM-DD.csv` with columns
//! `timestamp_secs,price,quantity`, sorted by timestamp.
//!
//! Timestamp units follow the frozen digit-length rule (13 -> ms, 16 -> us);
//! ambiguous lengths are rejected, never guessed. Empty or missing archives
//! are recorded as skipped days, mirroring the frozen loader's tolerance.
//!
//! No network. Additive parity tool; never regenerates frozen artifacts.

use amm_lab::pstt::cex::parse_aggtrade_line;
use amm_lab::pstt::error::PsttError;
use amm_lab::pstt::manifest::require_empty_output_dir;
use amm_lab::pstt::schema::AggTrade;
use clap::Parser;
use flate2::read::GzDecoder;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(about = "PSTT Binance aggTrades archive normalizer (no network)")]
struct Args {
    /// Directory containing daily archives for one symbol.
    #[arg(long)]
    input_dir: PathBuf,
    /// Symbol, e.g. ETHUSDC.
    #[arg(long)]
    symbol: String,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_nonempty_output: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn read_archive_lines(path: &Path) -> Result<Vec<String>, PsttError> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let mut lines = Vec::new();
    match ext {
        "zip" => {
            let file = File::open(path).map_err(|e| PsttError::io(path, e))?;
            let mut zf = zip::ZipArchive::new(file)
                .map_err(|e| PsttError::parse(format!("{}: {e}", path.display())))?;
            if zf.is_empty() {
                return Ok(lines);
            }
            // Frozen loader reads the first archive member only.
            let inner = zf
                .by_index(0)
                .map_err(|e| PsttError::parse(format!("{}: {e}", path.display())))?;
            for line in BufReader::new(inner).lines() {
                lines.push(line.map_err(|e| PsttError::io(path, e))?);
            }
        }
        "gz" => {
            let file = File::open(path).map_err(|e| PsttError::io(path, e))?;
            let reader: Box<dyn Read> = Box::new(GzDecoder::new(file));
            for line in BufReader::new(reader).lines() {
                lines.push(line.map_err(|e| PsttError::io(path, e))?);
            }
        }
        "csv" => {
            let file = File::open(path).map_err(|e| PsttError::io(path, e))?;
            for line in BufReader::new(file).lines() {
                lines.push(line.map_err(|e| PsttError::io(path, e))?);
            }
        }
        other => {
            return Err(PsttError::schema(format!(
                "unsupported archive extension .{other}: {}",
                path.display()
            )));
        }
    }
    Ok(lines)
}

fn run(args: Args) -> Result<(), PsttError> {
    require_empty_output_dir(&args.output_dir, args.allow_nonempty_output)?;
    let sym_dir = args.output_dir.join(&args.symbol);
    fs::create_dir_all(&sym_dir).map_err(|e| PsttError::io(&sym_dir, e))?;

    let prefix = format!("{}-aggTrades-", args.symbol);
    let mut archives: Vec<PathBuf> = fs::read_dir(&args.input_dir)
        .map_err(|e| PsttError::io(&args.input_dir, e))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(&prefix))
                .unwrap_or(false)
        })
        .collect();
    archives.sort();
    if archives.is_empty() {
        return Err(PsttError::invariant(format!(
            "no archives matching {prefix}* in {}",
            args.input_dir.display()
        )));
    }

    let mut days_written = Vec::new();
    let mut days_skipped = Vec::new();
    let mut total_rows = 0u64;
    for archive in &archives {
        let name = archive.file_name().and_then(|n| n.to_str()).unwrap_or("");
        // day = text between prefix and first extension dot
        let stem = name.strip_prefix(&prefix).unwrap_or(name);
        let day: String = stem.chars().take(10).collect();
        if day.len() != 10 {
            return Err(PsttError::schema(format!("cannot parse day from {name}")));
        }
        let meta_len = fs::metadata(archive).map(|m| m.len()).unwrap_or(0);
        if meta_len == 0 {
            days_skipped.push(day);
            continue;
        }
        let lines = read_archive_lines(archive)?;
        let mut rows: Vec<AggTrade> = Vec::with_capacity(lines.len());
        for line in &lines {
            // Header tolerance: skip a leading non-numeric header row.
            if line.starts_with("agg_trade_id") || line.starts_with("aggTradeId") {
                continue;
            }
            if let Some(row) = parse_aggtrade_line(line, false)? {
                rows.push(row);
            }
        }
        if rows.is_empty() {
            days_skipped.push(day);
            continue;
        }
        rows.sort_by(|a, b| a.timestamp_secs.total_cmp(&b.timestamp_secs));
        let out_path = sym_dir.join(format!("{}-aggTrades-{}.csv", args.symbol, day));
        let mut out = File::create(&out_path).map_err(|e| PsttError::io(&out_path, e))?;
        for r in &rows {
            // Re-emit in the frozen headerless column layout so downstream
            // loaders (`load_aggtrades_csv`) can consume the output directly.
            // Microsecond stamps (16 digits) preserve full precision for both
            // ms-sourced and us-sourced archives.
            writeln!(
                out,
                "0,{},{},0,0,{},true,true",
                r.price,
                r.quantity,
                (r.timestamp_secs * 1e6).round() as i64
            )
            .map_err(|e| PsttError::io(&out_path, e))?;
        }
        total_rows += rows.len() as u64;
        days_written.push(day);
    }

    let summary = serde_json::json!({
        "symbol": args.symbol,
        "archives": archives.len(),
        "days_written": days_written,
        "days_skipped": days_skipped,
        "rows": total_rows,
        "generator": "pstt_normalize_aggtrades",
        "note": "Additive Rust parity tool; not a frozen paper generator."
    });
    let summary_path = args.output_dir.join("pstt_aggtrades_summary.json");
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)
        .map_err(|e| PsttError::io(&summary_path, e))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
