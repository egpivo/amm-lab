use std::collections::HashMap;
use std::io::BufRead;

#[derive(Debug, Clone)]
struct QRow {
    gap: u8,
    vol: u8,
    flow: u8,
    _volume: u8,
    _prev_fee: u8,
    fee_bps: f64,
    q_value: f64,
    visits: u32,
}

fn load_q_table(path: &str) -> Vec<QRow> {
    let f = std::fs::File::open(path).expect("open q-table");
    let reader = std::io::BufReader::new(f);
    let mut rows = Vec::new();
    let mut lines = reader.lines();
    let _ = lines.next(); // skip header
    for line in lines.map_while(Result::ok) {
        let p: Vec<&str> = line.trim().split(',').collect();
        if p.len() < 6 {
            continue;
        }
        if p.len() >= 8 {
            rows.push(QRow {
                gap: p[0].parse().unwrap_or(0),
                vol: p[1].parse().unwrap_or(0),
                flow: p[2].parse().unwrap_or(0),
                _volume: p[3].parse().unwrap_or(0),
                _prev_fee: p[4].parse().unwrap_or(0),
                fee_bps: p[5].parse().unwrap_or(0.0),
                q_value: p[6].parse().unwrap_or(0.0),
                visits: p[7].parse().unwrap_or(0),
            });
        } else {
            rows.push(QRow {
                gap: p[0].parse().unwrap_or(0),
                vol: p[1].parse().unwrap_or(0),
                flow: p[2].parse().unwrap_or(0),
                _volume: 0,
                _prev_fee: 0,
                fee_bps: p[3].parse().unwrap_or(0.0),
                q_value: p[4].parse().unwrap_or(0.0),
                visits: p[5].parse().unwrap_or(0),
            });
        }
    }
    rows
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("data/processed/campbell_rl_fee_table.csv");
    let rows = load_q_table(path);
    println!("Q-table: {}  ({} rows)", path, rows.len());

    // ── Marginalize over volume_bucket and prev_fee_bucket ────────────────────
    // Group by (gap, vol, flow): collect all (fee_bps, q_value, visits) entries.
    type StateActions = HashMap<(u8, u8, u8), Vec<(f64, f64, u32)>>;
    let mut groups: StateActions = HashMap::new();
    for r in &rows {
        groups
            .entry((r.gap, r.vol, r.flow))
            .or_default()
            .push((r.fee_bps, r.q_value, r.visits));
    }

    // Per (gap, vol, flow): aggregate by fee_bps across (volume, prev_fee).
    //   - visits_total: sum across all (vol, pf) combos
    //   - q_agg: weighted average Q (weighted by visit count)
    // Also compute: best_action_by_visits (most visited) and best_action_by_q (argmax Q).
    struct AggState {
        best_fee_by_visits: f64,
        best_fee_by_q: f64,
        total_visits: u32,
        q_values: Vec<(f64, f64)>, // (fee_bps, q_value) sorted by fee_bps
    }

    let mut agg: HashMap<(u8, u8, u8), AggState> = HashMap::new();

    for (&state, actions) in &groups {
        // Aggregate by fee_bps: sum visits, weighted-avg Q.
        let mut by_fee: HashMap<u32, (f64, u32)> = HashMap::new(); // key=fee*100, val=(q_sum,visits)
        for &(fee, q, vis) in actions {
            let key = (fee * 100.0).round() as u32;
            let e = by_fee.entry(key).or_insert((0.0, 0));
            e.0 += q * vis as f64; // weighted q sum
            e.1 += vis;
        }
        let total_visits: u32 = by_fee.values().map(|(_, v)| v).sum();
        // Convert to (fee_bps, avg_q, visits) and sort by fee
        let mut fee_entries: Vec<(f64, f64, u32)> = by_fee
            .iter()
            .map(|(&k, &(q_sum, v))| {
                let fee = k as f64 / 100.0;
                let avg_q = if v > 0 { q_sum / v as f64 } else { 0.0 };
                (fee, avg_q, v)
            })
            .collect();
        fee_entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // best by visits
        let best_fee_by_visits = fee_entries
            .iter()
            .max_by_key(|&&(_, _, v)| v)
            .map(|&(f, _, _)| f)
            .unwrap_or(6.0);

        // best by Q (argmax Q)
        let best_fee_by_q = fee_entries
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|&(f, _, _)| f)
            .unwrap_or(6.0);

        let q_values: Vec<(f64, f64)> = fee_entries.iter().map(|&(f, q, _)| (f, q)).collect();

        agg.insert(
            state,
            AggState {
                best_fee_by_visits,
                best_fee_by_q,
                total_visits,
                q_values,
            },
        );
    }

    // ── Section 1: Policy table (argmax Q) ───────────────────────────────────
    println!("\n=== Policy Table (argmax Q, marginalized over volume/prev_fee) ===");
    println!("Note: best_q = what the inference policy chooses; best_vis = most-explored.");
    println!();

    let mut vf_pairs: Vec<(u8, u8)> = agg.keys().map(|&(_, v, f)| (v, f)).collect();
    vf_pairs.sort();
    vf_pairs.dedup();

    let action_bps = [3.0f64, 6.0, 10.0, 20.0, 30.0];
    let action_hdr: Vec<String> = action_bps
        .iter()
        .map(|b| format!("{:>6.0}bps", b))
        .collect();

    for (vol, flow) in &vf_pairs {
        let flow_label = match flow {
            0 => "sparse",
            1 => "arb-dom",
            2 => "fund-dom",
            _ => "mixed",
        };
        let vol_label = match vol {
            0 => "low-vol",
            1 => "mid-vol",
            _ => "hi-vol",
        };
        println!(
            "  vol={} ({}) | flow={} ({}):",
            vol, vol_label, flow, flow_label
        );
        println!(
            "  gap | best_q | best_vis | visits | {}",
            action_hdr.join(" ")
        );
        for gap in 0u8..6 {
            let key = (gap, *vol, *flow);
            if let Some(s) = agg.get(&key) {
                let q_strs: Vec<String> = action_bps
                    .iter()
                    .map(|&b| {
                        let q = s
                            .q_values
                            .iter()
                            .find(|&&(f, _)| (f - b).abs() < 0.05)
                            .map(|&(_, q)| q)
                            .unwrap_or(f64::NAN);
                        if q.is_nan() {
                            format!("{:>7}", "n/a")
                        } else {
                            format!("{:>7.4}", q)
                        }
                    })
                    .collect();
                println!(
                    "    {} |  {:5.1} |    {:5.1} | {:>6} | {}",
                    gap,
                    s.best_fee_by_q,
                    s.best_fee_by_visits,
                    s.total_visits,
                    q_strs.join(" "),
                );
            }
        }
        println!();
    }

    // ── Section 2: Monotonicity violations (argmax Q) ────────────────────────
    println!("=== Monotonicity Violations (argmax Q: fee(gap+1) < fee(gap)) ===");
    let mut total_transitions: u64 = 0;
    let mut total_violations: u64 = 0;
    let mut weighted_violations: u64 = 0;
    let mut weighted_total: u64 = 0;

    for (vol, flow) in &vf_pairs {
        let mut gap_rows: Vec<(u8, f64, u32)> = (0u8..6)
            .filter_map(|g| {
                agg.get(&(g, *vol, *flow))
                    .map(|s| (g, s.best_fee_by_q, s.total_visits))
            })
            .collect();
        gap_rows.sort_by_key(|&(g, _, _)| g);

        for i in 0..gap_rows.len().saturating_sub(1) {
            let (g0, fee0, vis0) = gap_rows[i];
            let (g1, fee1, vis1) = gap_rows[i + 1];
            let w = (vis0 + vis1) as u64;
            total_transitions += 1;
            weighted_total += w;
            if fee1 < fee0 - 0.01 {
                total_violations += 1;
                weighted_violations += w;
                println!(
                    "  vol={} flow={}: gap {}→{} fee {:.1}→{:.1} bps  (weight {})",
                    vol, flow, g0, g1, fee0, fee1, w
                );
            }
        }
    }
    println!(
        "\nViolations: {}/{} transitions ({:.1}%)",
        total_violations,
        total_transitions,
        if total_transitions > 0 {
            100.0 * total_violations as f64 / total_transitions as f64
        } else {
            0.0
        }
    );
    if weighted_total > 0 {
        println!(
            "Weighted violation rate: {:.1}%",
            100.0 * weighted_violations as f64 / weighted_total as f64
        );
    }

    // ── Section 3: Action distribution ───────────────────────────────────────
    println!("\n=== Action Distribution (fraction of states choosing each fee) ===");
    let n_states = agg.len() as f64;
    let mut action_counts: HashMap<u32, u32> = HashMap::new();
    for s in agg.values() {
        let key = (s.best_fee_by_q * 100.0).round() as u32;
        *action_counts.entry(key).or_insert(0) += 1;
    }
    let mut action_list: Vec<(f64, u32)> = action_counts
        .iter()
        .map(|(&k, &v)| (k as f64 / 100.0, v))
        .collect();
    action_list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (bps, cnt) in &action_list {
        println!(
            "  {:5.1} bps: {:>5} states ({:.1}%)",
            bps,
            cnt,
            100.0 * *cnt as f64 / n_states
        );
    }

    // ── Section 4: Monotone repair (argmax Q) ────────────────────────────────
    println!("\n=== Monotone Repair (raw vs repaired, argmax Q) ===");
    for (vol, flow) in &vf_pairs {
        let mut gap_rows: Vec<(u8, f64, u32)> = (0u8..6)
            .filter_map(|g| {
                agg.get(&(g, *vol, *flow))
                    .map(|s| (g, s.best_fee_by_q, s.total_visits))
            })
            .collect();
        gap_rows.sort_by_key(|&(g, _, _)| g);
        if gap_rows.len() < 2 {
            continue;
        }
        let mut repaired = vec![gap_rows[0].1];
        for i in 1..gap_rows.len() {
            repaired.push(repaired[i - 1].max(gap_rows[i].1));
        }
        let changed = gap_rows
            .iter()
            .zip(&repaired)
            .any(|((_, f, _), r)| (f - r).abs() > 0.01);
        if changed {
            let flow_label = match flow {
                0 => "sparse",
                1 => "arb-dom",
                2 => "fund-dom",
                _ => "mixed",
            };
            println!("\n  vol={} flow={} ({}):", vol, flow, flow_label);
            println!("  gap | raw  | repaired");
            for (i, &(g, raw, _)) in gap_rows.iter().enumerate() {
                let marker = if (raw - repaired[i]).abs() > 0.01 {
                    " <-- fixed"
                } else {
                    ""
                };
                println!("    {} | {:5.1} | {:5.1}{}", g, raw, repaired[i], marker);
            }
        }
    }
    println!();
}
