//! Thin standalone application orchestrator (frozen `build_m6_public.py`
//! stage-2 shape): weekly primitives JSON -> per-pool signed regions over the
//! frozen r_bar grid, synchronized ranking contrast, identification status,
//! and reference-robustness claim table.
//!
//! Bootstrap draws come from EITHER:
//!  - `--index-file`: an externally serialized index matrix per (label, ref)
//!    and for ranking (exact replay of a recorded schedule), OR
//!  - `--seed`: deterministic Rust-side generation (rand StdRng). This is
//!    explicitly NOT the historical NumPy stream: the frozen run derived its
//!    seeds through Python's salted `hash(...)` with no recorded
//!    PYTHONHASHSEED, so bitwise draw parity is unrecoverable. Results from
//!    `--seed` are a new draw schedule, never a reproduction of history.
//!
//! No network. Additive parity tool; never regenerates frozen artifacts and
//! never overwrites the frozen historical verdict.

use amm_lab::pstt::application::{
    WeeklyRecord, derive_seed, frozen_r_bar_grid, pool_signed_regions, ranking_signed,
    status_from_str,
};
use amm_lab::pstt::bootstrap::{IndexSchedule, block_length};
use amm_lab::pstt::classification::reference_robustness;
use amm_lab::pstt::error::PsttError;
use amm_lab::pstt::manifest::require_empty_output_dir;
use clap::Parser;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(about = "PSTT standalone application orchestrator (no network)")]
struct Args {
    /// Weekly primitives JSON: {label: {"last_trade": [rows], "vwap1s": [rows]}}
    /// with rows shaped like the frozen m6_public_weekly.json records.
    #[arg(long)]
    weekly_json: PathBuf,
    /// Label of the lower-fee pool (contrast lower arm), e.g. 5bp.
    #[arg(long)]
    lower_label: String,
    /// Label of the higher-fee pool (contrast higher arm), e.g. 30bp.
    #[arg(long)]
    higher_label: String,
    /// Bootstrap draws per region.
    #[arg(long, default_value_t = 1000)]
    draws: usize,
    /// Deterministic Rust seed (NOT NumPy-parity; see module docs).
    #[arg(long)]
    seed: Option<u64>,
    /// JSON file {"key": [[indices per draw], ...]} with keys
    /// "{label}:{reference}" and "rank:{reference}". Overrides --seed.
    #[arg(long)]
    index_file: Option<PathBuf>,
    #[arg(long, default_value_t = 0.95)]
    nominal: f64,
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

type WeeklyTable = BTreeMap<String, BTreeMap<String, Vec<WeeklyRecord>>>;

fn schedule_for(
    key: &str,
    n_weeks: usize,
    draws: usize,
    seed: Option<u64>,
    index_matrices: &Option<BTreeMap<String, Vec<Vec<usize>>>>,
) -> Result<IndexSchedule, PsttError> {
    if let Some(matrices) = index_matrices {
        let m = matrices.get(key).ok_or_else(|| {
            PsttError::MissingJoin(format!("index file has no schedule for key {key}"))
        })?;
        return IndexSchedule::from_matrix(m.clone(), n_weeks);
    }
    let base =
        seed.ok_or_else(|| PsttError::invariant("either --seed or --index-file must be provided"))?;
    let (label, reference) = key
        .split_once(':')
        .ok_or_else(|| PsttError::invariant(format!("bad schedule key {key}")))?;
    IndexSchedule::generate(
        derive_seed(base, label, reference),
        draws,
        n_weeks,
        block_length(n_weeks),
    )
}

fn run(args: Args) -> Result<(), PsttError> {
    require_empty_output_dir(&args.output_dir, args.allow_nonempty_output)?;
    let raw =
        fs::read_to_string(&args.weekly_json).map_err(|e| PsttError::io(&args.weekly_json, e))?;
    let table: WeeklyTable = serde_json::from_str(&raw)?;
    for label in [&args.lower_label, &args.higher_label] {
        if !table.contains_key(label) {
            return Err(PsttError::MissingJoin(format!(
                "weekly json has no label {label}"
            )));
        }
    }
    let index_matrices: Option<BTreeMap<String, Vec<Vec<usize>>>> = match &args.index_file {
        Some(p) => {
            let raw = fs::read_to_string(p).map_err(|e| PsttError::io(p, e))?;
            Some(serde_json::from_str(&raw)?)
        }
        None => None,
    };

    let references = ["last_trade", "vwap1s"];
    let grid = frozen_r_bar_grid();
    let mut pools_out = serde_json::Map::new();
    for (label, refs) in &table {
        let mut per_ref = serde_json::Map::new();
        for reference in references {
            let Some(records) = refs.get(reference) else {
                continue;
            };
            if records.is_empty() {
                continue;
            }
            let key = format!("{label}:{reference}");
            let sched = schedule_for(&key, records.len(), args.draws, args.seed, &index_matrices)?;
            let inf = pool_signed_regions(records, &sched, &grid, args.nominal)?;
            per_ref.insert(reference.to_string(), serde_json::to_value(&inf)?);
        }
        pools_out.insert(label.clone(), serde_json::Value::Object(per_ref));
    }

    // Synchronized ranking per reference on common weeks.
    let mut ranking_out = serde_json::Map::new();
    let mut status_by_ref: BTreeMap<&str, Option<String>> = BTreeMap::new();
    for reference in references {
        let lower = table[&args.lower_label].get(reference);
        let higher = table[&args.higher_label].get(reference);
        let (Some(lower), Some(higher)) = (lower, higher) else {
            status_by_ref.insert(reference, None);
            continue;
        };
        let common: usize = {
            let lw: std::collections::BTreeSet<&str> =
                lower.iter().map(|r| r.week.as_str()).collect();
            higher
                .iter()
                .filter(|r| lw.contains(r.week.as_str()))
                .count()
        };
        if common < 8 {
            status_by_ref.insert(reference, None);
            continue;
        }
        let key = format!("rank:{reference}");
        let sched = schedule_for(&key, common, args.draws, args.seed, &index_matrices)?;
        let rank = ranking_signed(lower, higher, &sched, &grid, args.nominal)?;
        status_by_ref.insert(reference, Some(rank.status.clone()));
        ranking_out.insert(reference.to_string(), serde_json::to_value(&rank)?);
    }

    let st_primary = status_by_ref
        .get("last_trade")
        .cloned()
        .flatten()
        .and_then(|s| status_from_str(&s));
    let st_robust = status_by_ref
        .get("vwap1s")
        .cloned()
        .flatten()
        .and_then(|s| status_from_str(&s));
    let robustness = reference_robustness(st_primary, st_robust);

    let manifest = serde_json::json!({
        "generator": "pstt_build_application",
        "provenance": {
            "authority": "frozen .local/selection_paper Python outputs (m6_public_manifest.json)",
            "role": "additive Rust parity/reconstruction implementation",
            "bootstrap": if args.index_file.is_some() {
                "externally serialized index schedule (exact replay)"
            } else {
                "new deterministic Rust draw schedule (NOT the historical NumPy stream)"
            },
        },
        "nominal": args.nominal,
        "bootstrap_draws": args.draws,
        "r_bar_grid": grid,
        "pools": pools_out,
        "ranking": ranking_out,
        "claim_table": [{
            "contrast": format!("Delta ({} vs {})", args.lower_label, args.higher_label),
            "primary_last_trade": status_by_ref.get("last_trade").cloned().flatten(),
            "robustness_vwap1s": status_by_ref.get("vwap1s").cloned().flatten(),
            "reference_robustness": robustness.as_str(),
        }],
    });
    let out_path = args.output_dir.join("pstt_application_manifest.json");
    fs::write(&out_path, serde_json::to_vec_pretty(&manifest)?)
        .map_err(|e| PsttError::io(&out_path, e))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&manifest["claim_table"])?
    );
    Ok(())
}
