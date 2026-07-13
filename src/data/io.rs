//! Load the reconstruction inputs (gzip events + JSON/CSV metadata) that
//! `build_outcomes.py` reads out of SQLite, into the in-memory form [`reconstruct`] wants.
//!
//! Parity-relevant choices mirror the Python loader exactly:
//! - rows are deduplicated on `(tx_hash, log_index)` keeping the **first** occurrence
//!   (SQLite `INSERT OR IGNORE` on `UNIQUE(tx_hash, log_index)`);
//! - reorg-`removed` rows are **not** filtered (the reconstruction SELECT ignores the flag);
//! - a pool's role/tier/tickbook default to `Unknown`/`0`/empty when absent, and token
//!   decimals default to 18 (handled in [`reconstruct`]'s `scale`);
//! - **every** distinct pool in the events file is reconstructed (`SELECT DISTINCT pool`),
//!   not just the frozen unit set.
//!
//! Per-tick liquidity in `ckpt_tickbook.json` routinely exceeds `u64` for stable pools
//! (observed up to ~3.6e27), so it is parsed via serde_json's arbitrary-precision `Number`
//! into `i128` rather than through the default `u64`/`f64` path, which would be lossy.

use crate::data::panel::{Panel, UnitRole};
use crate::data::reconstruct::{Event, reconstruct_pool};
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Removes a directory tree when dropped, so streaming shard temp files are cleaned up on
/// success, error, and panic alike (not only the happy path).
struct DirGuard(std::path::PathBuf);
impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("csv: {0}")]
    Csv(#[from] csv::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tick key {0:?} is not an i32: {1}")]
    TickKey(String, std::num::ParseIntError),
    #[error("tick liquidity {0:?} is not an i128: {1}")]
    TickLiq(String, std::num::ParseIntError),
    #[error("events header must begin with the `pool` column, got: {0:?}")]
    BadHeader(String),
}

/// One raw events row. Fields map to `events.csv.gz` headers by name (csv + serde);
/// unused columns (`unit_role`, `sqrtP`, `removed`) are simply not declared. Role comes
/// from `panel_units.json`, not this file's `unit_role` column, matching Python.
#[derive(Debug, Deserialize)]
struct RawRow {
    pool: String,
    tx_hash: String,
    block: i64,
    tx_index: i64,
    log_index: i64,
    ts: Option<i64>,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    owner: String,
    #[serde(rename = "tickLower")]
    tick_lower: Option<i32>,
    #[serde(rename = "tickUpper")]
    tick_upper: Option<i32>,
    #[serde(default)]
    liquidity_delta: String,
    #[serde(default)]
    swap_liquidity: String,
    #[serde(default)]
    amount0: String,
    #[serde(default)]
    amount1: String,
    tick: Option<i32>,
    #[serde(default)]
    token0: String,
    #[serde(default)]
    token1: String,
}

/// `(pool, tx_hash, Event)`; returns `None` for rows Python drops (`ts is None`).
fn to_event(r: RawRow) -> Option<(String, String, Event)> {
    let ts = r.ts?; // Python: `if ts is None: continue`
    // Python: `int(ldelta) if ldelta not in ("",None) else 0`.
    let ld = r.liquidity_delta.trim().parse::<i128>().unwrap_or(0);
    // Python: `cur_L=int(swl) if swl not in ("",None) else cur_L` -> None means carry-forward.
    let swl = {
        let s = r.swap_liquidity.trim();
        if s.is_empty() {
            None
        } else {
            s.parse::<i128>().ok()
        }
    };
    let ev = Event {
        ts,
        block: r.block,
        tx_index: r.tx_index,
        log_index: r.log_index,
        kind: r.kind,
        owner: r.owner,
        tick_lower: r.tick_lower,
        tick_upper: r.tick_upper,
        liquidity_delta: ld,
        swap_liquidity: swl,
        amount0: r.amount0,
        amount1: r.amount1,
        tick: r.tick,
        token0: r.token0,
        token1: r.token1,
    };
    Some((r.pool, r.tx_hash, ev))
}

/// Dedup on `(tx_hash, log_index)` (keep first), group by pool, over any reader whose bytes
/// are the events CSV with a header row.
fn read_events_from<R: io::Read>(inner: R) -> Result<HashMap<String, Vec<Event>>, IoError> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(inner);
    let mut seen: HashSet<(String, i64)> = HashSet::new();
    let mut by_pool: HashMap<String, Vec<Event>> = HashMap::new();
    for row in rdr.deserialize() {
        let raw: RawRow = row?;
        let Some((pool, tx_hash, ev)) = to_event(raw) else {
            continue;
        };
        if !seen.insert((tx_hash, ev.log_index)) {
            continue; // duplicate (tx_hash, log_index): OR IGNORE keeps the first
        }
        by_pool.entry(pool).or_default().push(ev);
    }
    Ok(by_pool)
}

/// Read gzipped `events.csv.gz`, dedup on `(tx_hash, log_index)` (keep first), group by pool.
pub fn read_events(path: &Path) -> Result<HashMap<String, Vec<Event>>, IoError> {
    read_events_from(GzDecoder::new(File::open(path)?))
}

/// Same as [`read_events`] for an uncompressed CSV (the streaming shard files).
fn read_events_plain(path: &Path) -> Result<HashMap<String, Vec<Event>>, IoError> {
    read_events_from(File::open(path)?)
}

/// The role lists in `panel_units.json`. `treated_main` is a superset and is intentionally
/// unmapped (Python does the same); later lists overwrite earlier on overlap.
#[derive(Debug, Deserialize)]
struct PanelUnits {
    #[serde(default)]
    treated_matched: Vec<String>,
    #[serde(default)]
    controls: Vec<String>,
    #[serde(default)]
    unmatched_treated: Vec<String>,
    #[serde(default)]
    crossvenue_forks: Vec<String>,
}

/// Pool -> role from `panel_units.json`.
pub fn load_roles(path: &Path) -> Result<HashMap<String, UnitRole>, IoError> {
    let pu: PanelUnits = serde_json::from_reader(File::open(path)?)?;
    let mut m = HashMap::new();
    for p in pu.treated_matched {
        m.insert(p, UnitRole::MatchedTreated);
    }
    for p in pu.controls {
        m.insert(p, UnitRole::MatchedControl);
    }
    for p in pu.unmatched_treated {
        m.insert(p, UnitRole::UnmatchedTreated);
    }
    for p in pu.crossvenue_forks {
        m.insert(p, UnitRole::CrossvenueFork);
    }
    Ok(m)
}

#[derive(Debug, Deserialize)]
struct TierRow {
    pool: String,
    tier: String,
}

/// Pool -> fee tier (ppm) from `feerev_panelvars.csv`; empty/`"0"` tiers are dropped
/// (Python: `if r["tier"] not in ("","0")`), so a missing pool defaults to 0 at the call site.
pub fn load_tiers(path: &Path) -> Result<HashMap<String, i64>, IoError> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut m = HashMap::new();
    for row in rdr.deserialize() {
        let r: TierRow = row?;
        let t = r.tier.trim();
        if t.is_empty() || t == "0" {
            continue;
        }
        if let Ok(v) = t.parse::<i64>() {
            m.insert(r.pool, v);
        }
    }
    Ok(m)
}

/// Token -> decimals from `token_decimals.json` (RPC-filled by `build_outcomes.py`).
pub fn load_decimals(path: &Path) -> Result<HashMap<String, u32>, IoError> {
    Ok(serde_json::from_reader(File::open(path)?)?)
}

/// Resolve the tickbook seed file: prefer `tickbook_init.json` (the finalized seed that
/// `build_outcomes.py` reads, so the Rust reconstruction uses the identical state), falling
/// back to the in-progress checkpoint `ckpt_tickbook.json` when the finalized file is absent.
fn tickbook_path(dir: &Path) -> std::path::PathBuf {
    let finalized = dir.join("tickbook_init.json");
    if finalized.exists() {
        finalized
    } else {
        dir.join("ckpt_tickbook.json")
    }
}

/// Pool -> (tick -> net liquidity) seed from `ckpt_tickbook.json`. Values are parsed as
/// `i128` via the arbitrary-precision literal (they exceed `u64` for stable pools).
pub fn load_tickbook(path: &Path) -> Result<HashMap<String, HashMap<i32, i128>>, IoError> {
    let raw: HashMap<String, HashMap<String, serde_json::Number>> =
        serde_json::from_reader(File::open(path)?)?;
    let mut out = HashMap::with_capacity(raw.len());
    for (pool, ticks) in raw {
        let mut m = HashMap::with_capacity(ticks.len());
        for (t, v) in ticks {
            let tick: i32 = t.parse().map_err(|e| IoError::TickKey(t.clone(), e))?;
            let s = v.as_str();
            let liq: i128 = s.parse().map_err(|e| IoError::TickLiq(s.to_string(), e))?;
            m.insert(tick, liq);
        }
        out.insert(pool, m);
    }
    Ok(out)
}

/// Everything [`reconstruct`] needs, assembled from a build_outcomes-style data directory.
pub struct ReconstructInputs {
    /// `(pool, role, tier, events)` for every distinct pool in the events file.
    pub pools: Vec<(String, UnitRole, i64, Vec<Event>)>,
    pub tickbook: HashMap<String, HashMap<i32, i128>>,
    pub decimals: HashMap<String, u32>,
}

/// Load all inputs from a directory laid out like `.local/amm_paper_c/data`
/// (`events/events.csv.gz`, `panel_units.json`, `feerev_panelvars.csv`,
/// `token_decimals.json`, `ckpt_tickbook.json`).
pub fn load_inputs(dir: &Path) -> Result<ReconstructInputs, IoError> {
    let events = read_events(&dir.join("events").join("events.csv.gz"))?;
    let roles = load_roles(&dir.join("panel_units.json"))?;
    let tiers = load_tiers(&dir.join("feerev_panelvars.csv"))?;
    // token_decimals.json is RPC-filled by build_outcomes; if absent (e.g. a smoke run on
    // raw events), default every token to 18 -- matching Python's `dec.get(tok,18)` fallback.
    // A real parity run must have it, or non-18 tokens (USDC=6, WBTC=8, ...) will mismatch.
    let decimals = match load_decimals(&dir.join("token_decimals.json")) {
        Ok(d) => d,
        Err(IoError::Io(e)) if e.kind() == io::ErrorKind::NotFound => {
            eprintln!(
                "warning: token_decimals.json not found -> defaulting ALL decimals to 18 \
                 (fine for a smoke run; a parity run will mismatch non-18 tokens)"
            );
            HashMap::new()
        }
        Err(e) => return Err(e),
    };
    let tickbook = load_tickbook(&tickbook_path(dir))?;

    let mut pools: Vec<_> = events
        .into_iter()
        .map(|(pool, evs)| {
            let role = roles.get(&pool).copied().unwrap_or(UnitRole::Unknown);
            let tier = tiers.get(&pool).copied().unwrap_or(0);
            (pool, role, tier, evs)
        })
        .collect();
    // Stable pool order (HashMap iteration is not) -> deterministic panel row order.
    pools.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(ReconstructInputs {
        pools,
        tickbook,
        decimals,
    })
}

/// FNV-1a over the pool string -> shard index. Only needs to be consistent within pass 1
/// (pass 2 reads every shard), so any deterministic distribution works.
fn shard_of(pool: &str, n_shards: usize) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in pool.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h % n_shards as u64) as usize
}

/// Out-of-core reconstruction: partition the events file into `n_shards` temp shards by
/// `hash(pool)`, then reconstruct one shard at a time. Peak memory is one shard's events
/// rather than the whole file, so this scales to the full pull (tens of millions of rows)
/// without holding it all in RAM.
///
/// Correctness vs the in-memory path: all rows for a pool land in the same shard, and
/// `(tx_hash, log_index)` is globally unique to one pool, so per-shard dedup keep-first (in
/// preserved stream order) is identical to the global `INSERT OR IGNORE`.
pub fn reconstruct_streaming(
    events_gz: &Path,
    roles: &HashMap<String, UnitRole>,
    tiers: &HashMap<String, i64>,
    decimals: &HashMap<String, u32>,
    tickbook: &HashMap<String, HashMap<i32, i128>>,
    n_shards: usize,
) -> Result<Panel, IoError> {
    assert!(n_shards >= 1, "n_shards must be >= 1");
    // Unique per invocation (pid + a process-wide counter) so concurrent calls -- including
    // parallel tests with the same n_shards -- never share a shard dir and delete each
    // other's files via DirGuard.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let shard_dir =
        std::env::temp_dir().join(format!("ammlab_shards_{}_{}", std::process::id(), seq));
    std::fs::create_dir_all(&shard_dir)?;
    let _guard = DirGuard(shard_dir.clone()); // cleans up on every exit path

    // ---- Pass 1: partition raw lines by hash(pool) into shard files (stream order kept) ----
    let mut rdr = BufReader::new(GzDecoder::new(File::open(events_gz)?));
    let mut header = String::new();
    rdr.read_line(&mut header)?;
    let header = header.trim_end().to_string();
    // Pass 1 finds the pool by splitting at the first comma, so the pool must be column 0.
    // The rest is parsed by header name in pass 2, so we write the *actual* source header
    // into each shard: if columns are ever reordered/added, the name-based parse stays
    // correct instead of silently reinterpreting positions under a stale fixed header.
    if !header.starts_with("pool,") {
        return Err(IoError::BadHeader(header));
    }
    let shard_path = |i: usize| shard_dir.join(format!("shard_{i}.csv"));
    let mut writers: Vec<BufWriter<File>> = Vec::with_capacity(n_shards);
    for i in 0..n_shards {
        let mut w = BufWriter::new(File::create(shard_path(i))?);
        writeln!(w, "{header}")?;
        writers.push(w);
    }

    let mut line = String::new();
    while rdr.read_line(&mut line)? > 0 {
        // The first CSV field is the pool address; fields are unquoted (hex/ints), so a
        // plain split at the first comma is safe and avoids a full parse in this pass.
        let pool = match line.split_once(',') {
            Some((p, _)) => p,
            None => {
                line.clear();
                continue;
            }
        };
        let w = &mut writers[shard_of(pool, n_shards)];
        w.write_all(line.as_bytes())?;
        if !line.ends_with('\n') {
            w.write_all(b"\n")?;
        }
        line.clear();
    }
    for w in &mut writers {
        w.flush()?;
    }
    drop(writers);

    // ---- Pass 2: reconstruct one shard at a time ----
    let empty = HashMap::new();
    let mut rows = Vec::new();
    for i in 0..n_shards {
        let by_pool = read_events_plain(&shard_path(i))?; // dedup + group within this shard
        for (pool, evs) in by_pool {
            let role = roles.get(&pool).copied().unwrap_or(UnitRole::Unknown);
            let tier = tiers.get(&pool).copied().unwrap_or(0);
            let seed = tickbook.get(&pool).unwrap_or(&empty);
            rows.extend(reconstruct_pool(&pool, role, tier, &evs, seed, decimals));
        }
    }
    rows.sort_by(|a, b| a.pool.cmp(&b.pool).then(a.week.cmp(&b.week)));

    Ok(Panel { rows }) // _guard drops here, removing the shard dir
}

/// Load the metadata (roles/tiers/decimals/tickbook) from a build_outcomes-style directory
/// and stream-reconstruct its `events/events.csv.gz` into a [`Panel`]. Memory-bounded
/// equivalent of `load_inputs` + `reconstruct`.
pub fn reconstruct_dir_streaming(dir: &Path, n_shards: usize) -> Result<Panel, IoError> {
    let roles = load_roles(&dir.join("panel_units.json"))?;
    let tiers = load_tiers(&dir.join("feerev_panelvars.csv"))?;
    let decimals = match load_decimals(&dir.join("token_decimals.json")) {
        Ok(d) => d,
        Err(IoError::Io(e)) if e.kind() == io::ErrorKind::NotFound => {
            eprintln!(
                "warning: token_decimals.json not found -> defaulting ALL decimals to 18 \
                 (fine for a smoke run; a parity run will mismatch non-18 tokens)"
            );
            HashMap::new()
        }
        Err(e) => return Err(e),
    };
    let tickbook = load_tickbook(&tickbook_path(dir))?;
    reconstruct_streaming(
        &dir.join("events").join("events.csv.gz"),
        &roles,
        &tiers,
        &decimals,
        &tickbook,
        n_shards,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use std::io::Write;

    const HEADER: &str = "pool,unit_role,tx_hash,block,tx_index,log_index,ts,type,owner,tickLower,tickUpper,liquidity_delta,swap_liquidity,amount0,amount1,sqrtP,tick,token0,token1,removed";

    fn write_gz_named(name: &str, rows: &[&str]) -> std::path::PathBuf {
        write_gz_hdr(name, HEADER, rows)
    }

    fn write_gz_hdr(name: &str, header: &str, rows: &[&str]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("iotest_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.csv.gz");
        let f = File::create(&path).unwrap();
        let mut enc = GzEncoder::new(f, Compression::default());
        writeln!(enc, "{header}").unwrap();
        for r in rows {
            writeln!(enc, "{r}").unwrap();
        }
        enc.finish().unwrap();
        path
    }

    #[test]
    fn streaming_rejects_header_without_pool_first() {
        // pool is not column 0 -> pass-1 comma-split would grab the wrong field; must fail loud.
        let bad = "block,pool,tx_hash,tx_index,log_index,ts,type,owner,tickLower,tickUpper,liquidity_delta,swap_liquidity,amount0,amount1,sqrtP,tick,token0,token1,removed";
        let rows = ["100,0xAAA,0xt1,0,0,1700000000,swap,,,,,10,1,2,0,0,0xt0,0xt1,0"];
        let path = write_gz_hdr("badhdr", bad, &rows);
        let err = reconstruct_streaming(
            &path,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            2,
        )
        .unwrap_err();
        assert!(
            matches!(err, IoError::BadHeader(_)),
            "expected BadHeader, got {err:?}"
        );
    }

    #[test]
    fn streaming_tolerates_reordered_trailing_columns() {
        // pool stays column 0 but the remaining columns are reordered; because shards carry
        // the real source header and pass 2 parses by name, reconstruction is unaffected.
        let hdr = "pool,tx_hash,type,owner,tickLower,tickUpper,liquidity_delta,swap_liquidity,amount0,amount1,tick,token0,token1,block,tx_index,log_index,ts,unit_role,sqrtP,removed";
        // fields ordered to match `hdr`
        let rows = [
            "0xAAA,0xt1,mint,0xlp,-300,300,1000,,0,0,,0xt0,0xt1,100,0,0,1700000000,matched_treated,0,0",
            "0xAAA,0xt2,swap,,,,,2000,10,20,0,0xt0,0xt1,101,0,1,1700003600,matched_treated,0,0",
        ];
        let path = write_gz_hdr("reordered", hdr, &rows);
        let panel = reconstruct_streaming(
            &path,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            2,
        )
        .unwrap();
        assert_eq!(panel.rows.len(), 1, "one pool-week");
        assert_eq!(panel.rows[0].swaps, 1);
        assert_eq!(panel.rows[0].lp_entry_count, 1);
        assert_eq!(panel.rows[0].net_liq, 1000);
    }

    #[test]
    fn dedup_removed_and_swap_liquidity_semantics() {
        // A mint, a swap with empty swap_liquidity, a duplicate of the mint by
        // (tx_hash, log_index) with different content, and a reorg-removed collect.
        let rows = [
            // mint: tickLower/Upper present, liquidity_delta=500, swap_liquidity empty
            "0xpool,matched_treated,0xtxA,100,1,0,1700000000,mint,0xlp,-100,100,500,,0,0,0,,0xt0,0xt1,0",
            // swap: empty swap_liquidity -> None (carry-forward), tick present
            "0xpool,matched_treated,0xtxB,101,0,1,1700000600,swap,,,,,,123,456,0,50,0xt0,0xt1,0",
            // duplicate (tx_hash=0xtxA, log_index=0): must be dropped, first kept
            "0xpool,matched_treated,0xtxA,100,1,0,1700000000,mint,0xEVIL,-1,1,999,,0,0,0,,0xt0,0xt1,0",
            // reorg-removed collect: removed=1 must still be INCLUDED (not filtered)
            "0xpool,matched_treated,0xtxC,102,0,0,1700001200,collect,0xlp,-100,100,0,,0,7,0,,0xt0,0xt1,1",
        ];
        let path = write_gz_named("dedup", &rows);
        let by_pool = read_events(&path).unwrap();
        let evs = &by_pool["0xpool"];
        assert_eq!(evs.len(), 3, "dup dropped, removed kept -> 3 rows");

        let mint = evs.iter().find(|e| e.kind == "mint").unwrap();
        assert_eq!(
            mint.liquidity_delta, 500,
            "first mint kept, not the 999 dup"
        );
        assert_eq!(mint.owner, "0xlp");
        assert_eq!(mint.swap_liquidity, None);

        let swap = evs.iter().find(|e| e.kind == "swap").unwrap();
        assert_eq!(
            swap.swap_liquidity, None,
            "empty swap_liquidity -> None (carry-forward)"
        );
        assert_eq!(swap.tick, Some(50));
        assert_eq!(swap.tick_lower, None);

        assert!(
            evs.iter().any(|e| e.kind == "collect"),
            "reorg-removed row must be kept"
        );
    }

    #[test]
    fn streaming_matches_in_memory_reconstruction() {
        use crate::data::panel::{Tol, compare};
        use crate::data::reconstruct::reconstruct;

        // Two pools, interleaved in stream order, incl a swap, mints, and a same-block JIT
        // burn, plus a duplicate (tx_hash,log_index) row that must be dropped keep-first.
        let rows = [
            "0xAAA,matched_treated,0xt1,100,0,0,1700000000,mint,0xlp,-300,300,1000,,0,0,0,,0xt0,0xt1,0",
            "0xBBB,matched_control,0xt2,100,1,0,1700000000,mint,0xlp2,-50,50,777,,0,0,0,,0xt0,0xt1,0",
            "0xAAA,matched_treated,0xt3,101,0,1,1700003600,swap,,,,,2000,10,20,0,0,0xt0,0xt1,0",
            "0xBBB,matched_control,0xt2,100,1,0,1700000000,mint,0xEVIL,-1,1,999,,0,0,0,,0xt0,0xt1,0", // dup
            "0xAAA,matched_treated,0xt4,102,0,0,1700007200,burn,0xlp,-300,300,500,,0,0,0,,0xt0,0xt1,0",
        ];
        let path = write_gz_named("stream", &rows);

        // in-memory reference
        let by_pool = read_events(&path).unwrap();
        let mem_pools: Vec<_> = by_pool
            .into_iter()
            .map(|(p, e)| {
                let role = if p == "0xAAA" {
                    UnitRole::MatchedTreated
                } else {
                    UnitRole::MatchedControl
                };
                (p, role, 3000i64, e)
            })
            .collect();
        let mem = reconstruct(&mem_pools, &HashMap::new(), &HashMap::new());

        // streaming
        let roles: HashMap<String, UnitRole> = [
            ("0xAAA".to_string(), UnitRole::MatchedTreated),
            ("0xBBB".to_string(), UnitRole::MatchedControl),
        ]
        .into_iter()
        .collect();
        let tiers: HashMap<String, i64> = [
            ("0xAAA".to_string(), 3000i64),
            ("0xBBB".to_string(), 3000i64),
        ]
        .into_iter()
        .collect();
        let stream = reconstruct_streaming(
            &path,
            &roles,
            &tiers,
            &HashMap::new(),
            &HashMap::new(),
            4, // multiple shards, both pools exercised
        )
        .unwrap();

        assert_eq!(mem.rows.len(), stream.rows.len(), "row count");
        let rep = compare(&mem, &stream, Tol::default());
        assert!(rep.is_pass(), "streaming != in-memory: {rep:?}");
    }

    #[test]
    fn tickbook_parses_values_beyond_u64() {
        let dir = std::env::temp_dir().join(format!("tbtest_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ckpt_tickbook.json");
        // A value well beyond u64 (1.8e19), as seen in real stable-pool seeds.
        std::fs::write(
            &path,
            r#"{"0xpool":{"-887220":3647317813095276252776910676,"100":-5}}"#,
        )
        .unwrap();
        let tb = load_tickbook(&path).unwrap();
        assert_eq!(tb["0xpool"][&-887220], 3647317813095276252776910676i128);
        assert_eq!(tb["0xpool"][&100], -5);
    }
}
