//! Build ISO-week signed-mark primitives from oriented fills + CSV aggTrades.
//!
//! No network. Requires an empty output directory unless explicitly overridden.

use amm_lab::pstt::asof::{strict_pre_timestamp, vwap_1s};
use amm_lab::pstt::block_time::{iso_week_label, panel_calendar_weeks};
use amm_lab::pstt::cex::load_aggtrades_csv;
use amm_lab::pstt::error::PsttError;
use amm_lab::pstt::manifest::require_empty_output_dir;
use amm_lab::pstt::marks::signed_mark;
use amm_lab::pstt::schema::ReferenceKind;
use amm_lab::pstt::weekly::{accumulate, bump_fill_count, empty_calendar, to_rows};
use clap::Parser;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(about = "PSTT weekly primitive builder (no network)")]
struct Args {
    /// CSV with columns: timestamp,pool,q,p_exec,direction,week,pair,fee,cex_symbol,invert
    #[arg(long)]
    fills_csv: PathBuf,
    /// Root containing `{SYMBOL}/{SYMBOL}-aggTrades-YYYY-MM-DD.csv`.
    #[arg(long)]
    trades_dir: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_nonempty_output: bool,
    /// Use the 104-slot panel calendar (default). Pass `--compact-calendar` to
    /// materialize only weeks present in the fill tape (for small fixtures).
    #[arg(long, default_value_t = false)]
    compact_calendar: bool,
}

#[derive(Debug, Deserialize)]
struct FillRow {
    timestamp: f64,
    pool: String,
    q: f64,
    p_exec: f64,
    direction: f64,
    week: String,
    pair: String,
    fee: u32,
    cex_symbol: String,
    invert: bool,
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
    let mut rdr = csv::Reader::from_path(&args.fills_csv)?;
    let mut fills: Vec<FillRow> = Vec::new();
    for row in rdr.deserialize() {
        fills.push(row?);
    }
    fills.sort_by(|a, b| {
        a.timestamp
            .total_cmp(&b.timestamp)
            .then_with(|| a.pool.cmp(&b.pool))
    });

    let mut pools = BTreeMap::<String, (String, u32)>::new();
    let mut invert_by_symbol = BTreeMap::<String, bool>::new();
    for f in &fills {
        pools.insert(f.pool.clone(), (f.pair.clone(), f.fee));
        invert_by_symbol.insert(f.cex_symbol.clone(), f.invert);
        let recomputed = iso_week_label(f.timestamp)?;
        if recomputed != f.week {
            return Err(PsttError::schema(format!(
                "fill week {} disagrees with ISO label {recomputed}",
                f.week
            )));
        }
    }
    let pool_list: Vec<String> = pools.keys().cloned().collect();
    let weeks = if args.compact_calendar {
        let mut w: Vec<String> = fills.iter().map(|f| f.week.clone()).collect();
        w.sort();
        w.dedup();
        w
    } else {
        panel_calendar_weeks()
    };
    let refs = [ReferenceKind::LastTrade, ReferenceKind::Vwap1s];
    let mut table = empty_calendar(&pool_list, &refs, &weeks);

    // Group fills by (symbol, UTC day) for reference loading.
    let mut by_day: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (i, f) in fills.iter().enumerate() {
        let day = amm_lab::pstt::block_time::utc_day(f.timestamp)?.to_string();
        by_day
            .entry((f.cex_symbol.clone(), day))
            .or_default()
            .push(i);
    }

    for ((symbol, day), idxs) in &by_day {
        let invert = *invert_by_symbol.get(symbol).unwrap_or(&false);
        let path = args
            .trades_dir
            .join(symbol)
            .join(format!("{symbol}-aggTrades-{day}.csv"));
        let trades = if path.exists() {
            load_aggtrades_csv(&path, invert)?
        } else {
            Vec::new()
        };
        for &i in idxs {
            let f = &fills[i];
            bump_fill_count(&mut table, &f.pool, &f.week, &refs)?;
            if let Ok(j) = strict_pre_timestamp(&trades, f.timestamp) {
                let m = signed_mark(f.direction, f.q, j.price, f.p_exec)?;
                accumulate(
                    &mut table,
                    &f.pool,
                    ReferenceKind::LastTrade,
                    &f.week,
                    m.ell,
                    f.q,
                )?;
            }
            if let Ok(j) = vwap_1s(&trades, f.timestamp) {
                let m = signed_mark(f.direction, f.q, j.price, f.p_exec)?;
                accumulate(
                    &mut table,
                    &f.pool,
                    ReferenceKind::Vwap1s,
                    &f.week,
                    m.ell,
                    f.q,
                )?;
            }
        }
    }

    let rows = to_rows(&table, &pools)?;
    let weekly_path = args.output_dir.join("pstt_weekly.json");
    fs::write(&weekly_path, serde_json::to_vec_pretty(&rows)?)
        .map_err(|e| PsttError::io(&weekly_path, e))?;

    let summary = serde_json::json!({
        "pools": pool_list.len(),
        "weeks": weeks.len(),
        "rows": rows.len(),
        "fills": fills.len(),
        "generator": "pstt_build_weekly_primitives",
        "note": "Additive Rust parity tool; not a frozen paper generator."
    });
    let summary_path = args.output_dir.join("pstt_weekly_summary.json");
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)
        .map_err(|e| PsttError::io(&summary_path, e))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
