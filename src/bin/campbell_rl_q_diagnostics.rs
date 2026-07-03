use std::collections::HashMap;
use std::io::BufRead;

#[derive(Debug, Clone)]
struct QRow {
    gap: u8,
    vol: u8,
    flow: u8,
    fee_bps: f64,
    q_value: f64,
    visits: u32,
}

fn load_q_table(path: &str) -> Vec<QRow> {
    let f = std::fs::File::open(path).expect("open q-table");
    let reader = std::io::BufReader::new(f);
    let mut rows = Vec::new();
    let mut lines = reader.lines();
    let _ = lines.next();

    for line in lines.map_while(Result::ok) {
        let p: Vec<&str> = line.trim().split(',').collect();
        if p.len() < 5 {
            continue;
        }
        rows.push(QRow {
            gap: p[0].parse().unwrap_or(0),
            vol: p[1].parse().unwrap_or(0),
            flow: p[2].parse().unwrap_or(0),
            fee_bps: p[3].parse().unwrap_or(0.0),
            q_value: p[4].parse().unwrap_or(0.0),
            visits: p[5].parse().unwrap_or(0),
        });
    }
    rows
}

fn main() {
    let rows = load_q_table("data/processed/campbell_rl_fee_table.csv");
    println!("Loaded {} rows", rows.len());

    // group by (gap, vol, flow), collect all (fee, q, visits)
    type StateActions = HashMap<(u8, u8, u8), Vec<(f64, f64, u32)>>;
    let mut groups: StateActions = HashMap::new();
    for r in &rows {
        groups
            .entry((r.gap, r.vol, r.flow))
            .or_default()
            .push((r.fee_bps, r.q_value, r.visits));
    }

    // for each state: pick action with highest q_value
    // best: (gap, vol, flow) -> (best_fee, visits_of_best_action, total_visits)
    let mut best: HashMap<(u8, u8, u8), (f64, u32)> = HashMap::new();
    for (&state, actions) in &groups {
        let (fee, _, visits) = actions
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        best.insert(state, (*fee, *visits));
    }

    // collect unique (vol, flow) pairs
    let mut vf_pairs: Vec<(u8, u8)> = best.keys().map(|&(_, v, f)| (v, f)).collect();
    vf_pairs.sort();
    vf_pairs.dedup();

    println!("Best fee by (vol_bucket, flow_bucket, gap_bucket)");
    for (vol, flow) in &vf_pairs {
        println!("\n vol={} flow={}:", vol, flow);
        println!("  gap | best_fee_bps | visits");
        for gap in 0u8..6 {
            if let Some(&(fee, vis)) = best.get(&(gap, *vol, *flow)) {
                println!("    {}  |   {:5.1}     | {}", gap, fee, vis);
            }
        }
    }

    // monotonicity violations
    println!("\n=== Monotonicity Violations (fee(gap+1) < fee(gap)) ===");
    let mut total_transitions: u64 = 0;
    let mut total_violations: u64 = 0;
    let mut weighted_violations: u64 = 0;
    let mut weighted_total: u64 = 0;

    for (vol, flow) in &vf_pairs {
        let mut gap_fees: Vec<(u8, f64, u32)> = (0u8..6)
            .filter_map(|g| best.get(&(g, *vol, *flow)).map(|&(fee, vis)| (g, fee, vis)))
            .collect();
        gap_fees.sort_by_key(|&(g, _, _)| g);

        for i in 0..gap_fees.len().saturating_sub(1) {
            let (g0, fee0, vis0) = gap_fees[i];
            let (g1, fee1, vis1) = gap_fees[i + 1];
            let w = (vis0 + vis1) as u64;
            total_transitions += 1;
            weighted_total += w;
            if fee1 < fee0 - 0.01 {
                total_violations += 1;
                weighted_violations += w;
                println!(
                    "  vol={} flow={}: gap {}->{} fee {:.1}->{:.1} bps (weight {})",
                    vol, flow, g0, g1, fee0, fee1, w
                );
            }
        }
    }
    println!(
        "\nViolations: {}/{} transitions ({:.1}%)",
        total_violations,
        total_transitions,
        100.0 * total_violations as f64 / total_transitions as f64
    );
    println!(
        "Weighted violation rate: {:.1}%",
        100.0 * weighted_violations as f64 / weighted_total as f64
    );

    // Monotone repair (isotonic projection, upward only)
    println!("\n=== Monotone Repair (raw vs repaired) ===");
    for (vol, flow) in &vf_pairs {
        let mut gap_fees: Vec<(u8, f64, u32)> = (0u8..6)
            .filter_map(|g| best.get(&(g, *vol, *flow)).map(|&(fee, vis)| (g, fee, vis)))
            .collect();
        gap_fees.sort_by_key(|&(g, _, _)| g);
        if gap_fees.len() < 2 {
            continue;
        }

        let mut repaired = vec![gap_fees[0].1];
        for i in 1..gap_fees.len() {
            repaired.push(repaired[i - 1].max(gap_fees[i].1));
        }

        let changed = gap_fees
            .iter()
            .zip(&repaired)
            .any(|((_, f, _), r)| (f - r).abs() > 0.01);
        if changed {
            println!("\n vol={} flow={}:", vol, flow);
            println!("  gap | raw  | repaired");
            for (i, &(g, raw, _)) in gap_fees.iter().enumerate() {
                let marker = if (raw - repaired[i]).abs() > 0.01 {
                    " <-- fixed"
                } else {
                    ""
                };
                println!("    {} | {:5.1} | {:5.1}{}", g, raw, repaired[i], marker);
            }
        }
    }
}
