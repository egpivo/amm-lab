//! Exact deep-subset z pipeline for the LVR paper (physical-clock, round 17).
//!
//! Rust replacement for the Python prototype `.local/lvr/tools/
//! deep_subset_z.py` (retained as spec). Streams the extracted
//! three-pool event file, replays each pool from its finalized
//! pre-window tickbook seed (crate `data::Book`), and computes for every
//! ELIGIBLE pre-treatment swap the exact fee-free curve input/output and
//! the directional epsilon-band depths, using `data::v3math` (integer
//! TickMath / SqrtPriceMath; no floating point before final z output).
//!
//! Validation layers per round 15/16 (abort or skip on failure): TickMath
//! containment of post-swap sqrt price; replayed active-L state parity;
//! fee-free output parity within a rounding envelope; fee-tier consistency;
//! then z only for swaps that pass all checks, with coverage reported.
//!
//! Initialization: tick liquidity from the pre-window tickbook (complete
//! Mint/Burn history); spot state via a disclosed one-swap warm-up.
//! Event order within each pool is ASSERTED nondecreasing in
//! (block, tx_index, log_index); any violation aborts.

use amm_lab::data::book::Book;
use amm_lab::data::v3math::{
    MAX_TICK, MIN_TICK, amount0_delta, amount1_delta, eps_boundary, get_sqrt_ratio_at_tick,
};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use ruint::aliases::U256;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

const BASE: &str = "/Users/joseph/amm-lab";
const EPS_BPS: [u32; 3] = [10, 50, 100];

fn pools() -> HashMap<String, u32> {
    HashMap::from([
        ("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640".into(), 500),
        ("0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8".into(), 3000),
        ("0xe0554a476a092703abdb3ef35c80e0d76d32939f".into(), 100),
    ])
}

fn u256_to_f64(v: U256) -> f64 {
    let bits = 256 - v.leading_zeros();
    if bits <= 100 {
        let x: u128 = v.try_into().unwrap();
        x as f64
    } else {
        let shift = bits - 100;
        let hi: u128 = (v >> shift).try_into().unwrap();
        hi as f64 * (shift as f64).exp2()
    }
}

/// Piecewise fee-free (input, output) from s_from to s_to over the tick
/// ladder. Input rounds up, output rounds down (SqrtPriceMath semantics).
fn piecewise_io(
    book: &mut Book,
    tick0: i32,
    l0: i128,
    s_from: U256,
    s_to: U256,
) -> (U256, U256, u32, i128) {
    if s_to == s_from {
        return (U256::ZERO, U256::ZERO, 0, l0);
    }
    let up = s_to > s_from;
    let mut liq = l0;
    let mut inp = U256::ZERO;
    let mut out = U256::ZERO;
    let mut ncross = 0u32;
    let mut s_cur = s_from;
    if up {
        for (t, net) in book.crossings(tick0, MAX_TICK) {
            let s_t = get_sqrt_ratio_at_tick(t);
            if s_t >= s_to {
                break;
            }
            if s_t > s_cur && liq > 0 {
                inp += amount1_delta(liq as u128, s_cur, s_t, true);
                out += amount0_delta(liq as u128, s_cur, s_t, false);
            }
            if s_t > s_cur {
                s_cur = s_t;
            }
            liq += net;
            ncross += 1;
        }
        if liq > 0 && s_to > s_cur {
            inp += amount1_delta(liq as u128, s_cur, s_to, true);
            out += amount0_delta(liq as u128, s_cur, s_to, false);
        }
    } else {
        for (t, net) in book.crossings(MIN_TICK - 1, tick0).into_iter().rev() {
            let s_t = get_sqrt_ratio_at_tick(t);
            if s_t <= s_to {
                break;
            }
            if s_t < s_cur && liq > 0 {
                inp += amount0_delta(liq as u128, s_t, s_cur, true);
                out += amount1_delta(liq as u128, s_t, s_cur, false);
            }
            if s_t < s_cur {
                s_cur = s_t;
            }
            liq -= net;
            ncross += 1;
        }
        if liq > 0 && s_to < s_cur {
            inp += amount0_delta(liq as u128, s_to, s_cur, true);
            out += amount1_delta(liq as u128, s_to, s_cur, false);
        }
    }
    (inp, out, ncross, liq)
}

#[derive(Default)]
struct PoolRun {
    tier: u32,
    book: Book,
    spot: Option<(U256, i32, i128)>,
    last_key: (u64, u64, u64),
    // stats
    swaps: u64,
    warmup: u64,
    zero_move: u64,
    missing_post: u64,
    l_parity_fail: u64,
    output_parity_fail: u64,
    fee_tier_fail: u64,
    eligible: u64,
    n_z: u64,
    band_cross: HashMap<u32, u64>,
    input_overflow: u64,
    boundary_landing: u64,
    // round-18 volume audit (f64 accumulation of i128 amounts is fine
    // for SHARE reporting; exact values live in the failures detail file)
    gross_in_vol_total: f64,
    gross_in_vol_eligible: f64,
    out_vol_total: f64,
    out_vol_eligible: f64,
}

fn treatment_cutoffs() -> HashMap<String, i64> {
    let mut out = HashMap::new();
    let f = File::open(format!("{BASE}/data/causality/setfeeprotocol_events.csv")).unwrap();
    let mut rdr = csv::Reader::from_reader(f);
    let hdr = rdr.headers().unwrap().clone();
    let ip = hdr.iter().position(|h| h == "pool").unwrap();
    let it = hdr.iter().position(|h| h == "timestamp").unwrap();
    let target = pools();
    for rec in rdr.records() {
        let rec = rec.unwrap();
        let p = rec[ip].to_lowercase();
        if target.contains_key(&p) {
            let ts: i64 = rec[it].parse().unwrap();
            out.entry(p)
                .and_modify(|v: &mut i64| *v = (*v).min(ts))
                .or_insert(ts);
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let short_window: u64 = args
        .iter()
        .position(|a| a == "--short-window")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let seeds: HashMap<String, HashMap<String, serde_json::Value>> = serde_json::from_reader(
        File::open(format!("{BASE}/.local/amm_paper_c/data/tickbook_init.json")).unwrap(),
    )
    .unwrap();
    let cuts = treatment_cutoffs();
    eprintln!("treatment cutoffs: {cuts:?}");

    let mut runs: HashMap<String, PoolRun> = HashMap::new();
    for (p, tier) in pools() {
        let mut book = Book::new();
        if let Some(seed) = seeds.get(&p) {
            for (t, v) in seed {
                let net: i128 = v.as_str().map(|s| s.parse().unwrap()).unwrap_or_else(|| {
                    v.as_i64()
                        .map(i128::from)
                        .unwrap_or_else(|| v.to_string().parse().unwrap())
                });
                book.apply(t.parse().unwrap(), net);
            }
        }
        runs.insert(
            p,
            PoolRun {
                tier,
                book,
                last_key: (0, 0, 0),
                ..Default::default()
            },
        );
    }

    let zpath = format!("{BASE}/.local/lvr/deep_subset_z_rows.csv.gz");
    let mut zout = GzEncoder::new(File::create(&zpath).unwrap(), Compression::default());
    writeln!(zout, "pool,ts,up,eps_bps,z").unwrap();
    // round-18: per-failure detail for the selection audit (offline
    // distributional analysis by size, tick movement, crossings, time)
    let fpath = format!("{BASE}/.local/lvr/deep_subset_failures.csv.gz");
    let mut fout = GzEncoder::new(File::create(&fpath).unwrap(), Compression::default());
    writeln!(
        fout,
        "pool,block,ts,reason,gross_in,event_out,tick_move,ncross"
    )
    .unwrap();

    let f = File::open(format!(
        "{BASE}/.local/lvr/deep_subset_3pools_all_events.csv.gz"
    ))
    .unwrap();
    let mut rdr = csv::Reader::from_reader(GzDecoder::new(f));
    let hdr = rdr.headers().unwrap().clone();
    let col = |n: &str| {
        hdr.iter()
            .position(|h| h == n)
            .unwrap_or_else(|| panic!("missing column {n}"))
    };
    let (c_pool, c_block, c_txi, c_logi, c_ts, c_type) = (
        col("pool"),
        col("block"),
        col("tx_index"),
        col("log_index"),
        col("ts"),
        col("type"),
    );
    let (c_tl, c_tu, c_liqd, c_swapl) = (
        col("tickLower"),
        col("tickUpper"),
        col("liquidity_delta"),
        col("swap_liquidity"),
    );
    let (c_a0, c_a1, c_sqrtp, c_tick) = (col("amount0"), col("amount1"), col("sqrtP"), col("tick"));

    let mut n_rows: u64 = 0;
    for rec in rdr.records() {
        let rec = rec.unwrap();
        let pool = rec[c_pool].to_lowercase();
        let Some(run) = runs.get_mut(&pool) else {
            continue;
        };
        if short_window > 0 && run.swaps >= short_window {
            continue;
        }
        n_rows += 1;
        let key = (
            rec[c_block].parse().unwrap_or(0),
            rec[c_txi].parse().unwrap_or(0),
            rec[c_logi].parse().unwrap_or(0),
        );
        assert!(
            key >= run.last_key,
            "event order violated for {pool} at block {:?} (have {:?})",
            key,
            run.last_key
        );
        run.last_key = key;

        match &rec[c_type] {
            "mint" | "burn" => {
                let (tl, tu, liq) = (&rec[c_tl], &rec[c_tu], &rec[c_liqd]);
                if !tl.is_empty() && !tu.is_empty() && !liq.is_empty() {
                    let sgn: i128 = if &rec[c_type] == "mint" { 1 } else { -1 };
                    let l: i128 = liq.parse().unwrap();
                    let (tl, tu): (i32, i32) = (tl.parse().unwrap(), tu.parse().unwrap());
                    run.book.apply(tl, sgn * l);
                    run.book.apply(tu, -sgn * l);
                    if let Some((s, tk, l0)) = run.spot
                        && tl <= tk
                        && tk < tu
                    {
                        run.spot = Some((s, tk, l0 + sgn * l));
                    }
                }
            }
            "swap" => {
                run.swaps += 1;
                if rec[c_sqrtp].is_empty() || rec[c_tick].is_empty() {
                    run.missing_post += 1;
                    continue;
                }
                let s_post = U256::from_str_radix(&rec[c_sqrtp], 10).unwrap();
                let tick_post: i32 = rec[c_tick].parse().unwrap();
                let l_post: Option<i128> = if rec[c_swapl].is_empty() {
                    None
                } else {
                    rec[c_swapl].parse().ok()
                };
                // layer 0: TickMath containment (aborts on violation).
                // v3 edge case: a zero-for-one swap landing EXACTLY on a
                // tick boundary sets slot0.tick = boundary - 1 while
                // sqrtP == sqrt(boundary), so the right edge is inclusive;
                // exact-boundary landings are counted separately.
                let s_lo = get_sqrt_ratio_at_tick(tick_post);
                let s_hi = get_sqrt_ratio_at_tick(tick_post + 1);
                assert!(
                    s_lo <= s_post && s_post <= s_hi,
                    "TickMath containment violated: pool {pool} block {} tick {tick_post}",
                    &rec[c_block]
                );
                if s_post == s_hi {
                    run.boundary_landing += 1;
                }
                let Some((s_pre, tick_pre, l_pre)) = run.spot else {
                    run.spot = Some((s_post, tick_post, l_post.unwrap_or(0)));
                    run.warmup += 1;
                    continue;
                };
                if s_post == s_pre {
                    run.zero_move += 1;
                    run.spot = Some((s_post, tick_post, l_post.unwrap_or(l_pre)));
                    continue;
                }

                let (i_curve, o_curve, ncross, l_end) =
                    piecewise_io(&mut run.book, tick_pre, l_pre, s_pre, s_post);
                let up = s_post > s_pre;
                let a0: i128 = rec[c_a0].parse().unwrap_or(0);
                let a1: i128 = rec[c_a1].parse().unwrap_or(0);
                let gross_in: i128 = if up { a1 } else { a0 };
                let event_out: i128 = if up { -a0 } else { -a1 };

                let state_ok = l_post.is_none_or(|lp| l_end == lp);
                if !state_ok {
                    run.l_parity_fail += 1;
                    eprintln!(
                        "L-PARITY FAIL {pool} block {} tx {} log {}: pre_tick {tick_pre} post_tick {tick_post} l_pre {l_pre} l_end {l_end} l_event {:?} ncross {ncross}",
                        &rec[c_block], &rec[c_txi], &rec[c_logi], l_post
                    );
                }
                let tol_out = 2 * (ncross as i128 + 2);
                let o_curve_i: i128 = match TryInto::<u128>::try_into(o_curve) {
                    Ok(v) if v <= i128::MAX as u128 => v as i128,
                    _ => {
                        run.input_overflow += 1;
                        -1
                    }
                };
                let output_ok =
                    event_out > 0 && o_curve_i >= 0 && (o_curve_i - event_out).abs() <= tol_out;
                if !output_ok {
                    run.output_parity_fail += 1;
                }
                let i_curve_i: i128 = match TryInto::<u128>::try_into(i_curve) {
                    Ok(v) if v <= i128::MAX as u128 => v as i128,
                    _ => {
                        run.input_overflow += 1;
                        -1
                    }
                };
                let mut fee_ok = false;
                if gross_in > 0 && i_curve_i > 0 {
                    let fee_recon = gross_in - i_curve_i;
                    let fee_expect = gross_in * run.tier as i128 / 1_000_000;
                    let tol_fee = (gross_in / 10_000).max((ncross as i128 + 2) * 10);
                    fee_ok = fee_recon >= 0 && (fee_recon - fee_expect).abs() <= tol_fee;
                }
                if !fee_ok {
                    run.fee_tier_fail += 1;
                }

                let eligible = state_ok && output_ok && fee_ok;
                let tick_move = (tick_post - tick_pre).abs();
                run.gross_in_vol_total += gross_in.max(0) as f64;
                run.out_vol_total += event_out.max(0) as f64;
                if eligible {
                    run.eligible += 1;
                    run.gross_in_vol_eligible += gross_in.max(0) as f64;
                    run.out_vol_eligible += event_out.max(0) as f64;
                } else {
                    let reason = if !state_ok {
                        "l_parity"
                    } else if !output_ok {
                        "output_parity"
                    } else {
                        "fee_tier"
                    };
                    writeln!(
                        fout,
                        "{pool},{},{},{reason},{gross_in},{event_out},{tick_move},{ncross}",
                        &rec[c_block], &rec[c_ts]
                    )
                    .unwrap();
                }
                let ts: i64 = rec[c_ts].parse().unwrap_or(i64::MAX);
                let in_window = ts < *cuts.get(&pool).unwrap_or(&i64::MAX);
                if run.tier != 100 && eligible && in_window && l_pre > 0 {
                    for eps in EPS_BPS {
                        let s_b = eps_boundary(s_pre, eps, up);
                        let (d_eps, _, ncr, _) =
                            piecewise_io(&mut run.book, tick_pre, l_pre, s_pre, s_b);
                        if d_eps > U256::ZERO {
                            let z = u256_to_f64(i_curve) / u256_to_f64(d_eps);
                            writeln!(zout, "{pool},{ts},{},{eps},{z:.6e}", u8::from(up)).unwrap();
                            if ncr > 0 {
                                *run.band_cross.entry(eps).or_insert(0) += 1;
                            }
                        }
                    }
                    run.n_z += 1;
                }
                run.spot = Some((s_post, tick_post, l_post.unwrap_or(l_end)));
            }
            _ => {}
        }
    }
    zout.finish().unwrap();
    fout.finish().unwrap();

    let manifest: serde_json::Value = serde_json::json!({
        "arithmetic_version": "v3math-exact-1",
        "rows_processed": n_rows,
        "eps_bps": EPS_BPS,
        "treatment_cutoffs": cuts,
        "short_window": short_window,
        "pools": runs.iter().map(|(p, r)| {
            (p.clone(), serde_json::json!({
                "tier": r.tier,
                "swaps": r.swaps,
                "warmup_dropped": r.warmup,
                "zero_move": r.zero_move,
                "missing_post_state": r.missing_post,
                "l_parity_fail": r.l_parity_fail,
                "output_parity_fail": r.output_parity_fail,
                "fee_tier_fail": r.fee_tier_fail,
                "eligible": r.eligible,
                "eligibility_coverage": r.eligible as f64 / (r.swaps.saturating_sub(r.warmup)).max(1) as f64,
                "z_swaps": r.n_z,
                "band_cross_share": r.band_cross.iter().map(|(e, n)| (e.to_string(), *n as f64 / r.n_z.max(1) as f64)).collect::<HashMap<_,_>>(),
                "amount_overflow": r.input_overflow,
                "boundary_landings": r.boundary_landing,
                "gross_in_volume_share_eligible": r.gross_in_vol_eligible / r.gross_in_vol_total.max(1.0),
                "output_volume_share_eligible": r.out_vol_eligible / r.out_vol_total.max(1.0),
            }))
        }).collect::<HashMap<_,_>>(),
    });
    let mpath = format!("{BASE}/.local/lvr/deep_subset_z_manifest.json");
    serde_json::to_writer_pretty(File::create(&mpath).unwrap(), &manifest).unwrap();
    println!("{}", serde_json::to_string_pretty(&manifest).unwrap());
}
