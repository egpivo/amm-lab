use std::collections::HashMap;
use std::fs;
use std::io::Write;

#[derive(Debug)]
struct PathRow {
    seed: u64,
    avg_fee_bps: f64,
    hedged_pnl: f64,
    lp_vs_hold: f64,
    fee_revenue: f64,
    lvr: f64,
}

#[derive(Debug)]
struct StepRow {
    oracle_gap_basis: f64,
    inventory_skew: f64,
    _recent_vol: f64,
    fee_bps: f64,
}

fn main() {
    let compare_path = "data/processed/campbell_sim_compare.csv";
    let step_path = "data/processed/campbell_dynamic_fee_steps.csv";
    let mut rdr = csv::Reader::from_path(compare_path).unwrap();
    let mut stdt = csv::Reader::from_path(step_path).unwrap();
    let mut by_policy: HashMap<String, Vec<PathRow>> = HashMap::new();
    let mut step_by_policy: HashMap<String, Vec<StepRow>> = HashMap::new();

    for result in rdr.records() {
        let rec = result.unwrap();
        let policy = rec[0].to_string();
        let row = PathRow {
            seed: rec[1].parse().unwrap(),
            avg_fee_bps: rec[2].parse().unwrap(),
            hedged_pnl: rec[3].parse().unwrap(),
            lp_vs_hold: rec[4].parse().unwrap(),
            fee_revenue: rec[5].parse().unwrap(),
            lvr: rec[6].parse().unwrap(),
        };
        by_policy.entry(policy).or_default().push(row);
    }
    eprintln!("policies: {:?}", by_policy.keys().collect::<Vec<_>>());
    eprintln!(
        "paths per policy: {}",
        by_policy.values().next().unwrap().len()
    );

    for result in stdt.records() {
        let rec = result.unwrap();
        let policy = rec[0].to_string();
        let row = StepRow {
            oracle_gap_basis: rec[4].parse().unwrap(),
            inventory_skew: rec[5].parse().unwrap(),
            _recent_vol: rec[6].parse().unwrap(),
            fee_bps: rec[7].parse().unwrap(),
        };
        step_by_policy.entry(policy).or_default().push(row);
    }
    eprintln!(
        "step rows per policy: {}",
        step_by_policy.values().next().unwrap().len()
    );

    for (policy, rows) in &by_policy {
        let mut pnl: Vec<f64> = rows.iter().map(|r| r.hedged_pnl).collect();
        pnl.sort_by(|a, b| a.partial_cmp(b).unwrap());

        eprintln!(
            "{}: mean={:.2} median={:.2} p05={:.2} p95={:.2}",
            policy,
            mean(&pnl),
            percentile(&pnl, 50.0),
            percentile(&pnl, 5.0),
            percentile(&pnl, 95.0),
        );
    }

    let dynamic = ["oracle_gap", "inventory_gap"];
    for policy in &dynamic {
        if let Some(rows) = step_by_policy.get(*policy) {
            let fee: Vec<f64> = rows.iter().map(|r| r.fee_bps).collect();
            let gap: Vec<f64> = rows.iter().map(|r| r.oracle_gap_basis.abs()).collect();
            let skew: Vec<f64> = rows.iter().map(|r| r.inventory_skew.abs()).collect();
            eprintln!(
                "{} corr(fee, |gap|)={:.4} corr(fee,|skew|)={:.4}",
                policy,
                pearson(&fee, &gap),
                pearson(&fee, &skew),
            );
        }
    }
    fs::create_dir_all("data/processed").unwrap();
    let audit_path = "data/processed/campbell_dynamic_fee_policy_audit.csv";
    let mut af = fs::File::create(audit_path).unwrap();
    writeln!(
        af,
        "policy,n_paths,mean_hedged_pnl,median_hedged_pnl,p05_hedged_pnl,p95_hedged_pnl,\
    mean_fee_bps,median_fee_bps,p05_fee_bps,p95_fee_bps,\
    corr_fee_bas_oracle_gap,corr_fee_abs_inventory_skew"
    )
    .unwrap();

    for policy in &["fixed_6bps", "fixed_10bps", "oracle_gap", "inventory_gap"] {
        let rows = &by_policy[*policy];
        let mut pnl: Vec<f64> = rows.iter().map(|r| r.hedged_pnl).collect();
        pnl.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut fee: Vec<f64> = rows.iter().map(|r| r.avg_fee_bps).collect();
        fee.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let (corr_gap, corr_skew) = if let Some(srows) = step_by_policy.get(*policy) {
            let sf: Vec<f64> = srows.iter().map(|r| r.fee_bps).collect();
            let sg: Vec<f64> = srows.iter().map(|r| r.oracle_gap_basis.abs()).collect();
            let ss: Vec<f64> = srows.iter().map(|r| r.inventory_skew.abs()).collect();
            (pearson(&sf, &sg), pearson(&sf, &ss))
        } else {
            (f64::NAN, f64::NAN)
        };

        writeln!(
            af,
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            policy,
            rows.len(),
            mean(&pnl),
            percentile(&pnl, 50.0),
            percentile(&pnl, 5.0),
            percentile(&pnl, 95.0),
            mean(&fee),
            percentile(&fee, 50.0),
            percentile(&fee, 5.0),
            percentile(&fee, 95.0),
            corr_gap,
            corr_skew,
        )
        .unwrap();
    }
    eprintln!("written: {audit_path}");

    let baseline: HashMap<u64, &PathRow> = by_policy["fixed_6bps"]
        .iter()
        .map(|r| (r.seed, r))
        .collect();
    let paired_path = "data/processed/campbell_dynamic_fee_paired_delta.csv";
    let mut pf = fs::File::create(paired_path).unwrap();
    writeln!(
        pf,
        "policy,seed,delta_hedged_pnl,delta_lp_vs_hold,delta_fee_revenue,delta_lvr"
    )
    .unwrap();

    for policy in &["oracle_gap", "inventory_gap", "fixed_10bps"] {
        for row in &by_policy[*policy] {
            let base = baseline[&row.seed];
            writeln!(
                pf,
                "{},{},{:.4},{:.4},{:.4},{:.4}",
                policy,
                row.seed,
                row.hedged_pnl - base.hedged_pnl,
                row.lp_vs_hold - base.lp_vs_hold,
                row.fee_revenue - base.fee_revenue,
                row.lvr - base.lvr
            )
            .unwrap();
        }
    }
    eprintln!("written: {paired_path}");

    let summary_path = "data/processed/campbell_dynamic_fee_paired_summary.csv";
    let mut sf2 = fs::File::create(summary_path).unwrap();
    writeln!(
        sf2,
        "policy,mean_delta_hedged_pnl,median_delta_hedged_pnl,p05_delta_hedged_pnl,\
    pct_paths_beating_fixed6,mean_delta_lp_vs_hold,median_delta_lp_vs_hold,\
    pct_paths_lp_vs_hold_beating_fixed6"
    )
    .unwrap();

    for policy in &["oracle_gap", "inventory_gap", "fixed_10bps"] {
        let rows = &by_policy[*policy];
        let mut dpnl: Vec<f64> = rows
            .iter()
            .map(|r| r.hedged_pnl - baseline[&r.seed].hedged_pnl)
            .collect();
        let mut dlvh: Vec<f64> = rows
            .iter()
            .map(|r| r.lp_vs_hold - baseline[&r.seed].lp_vs_hold)
            .collect();
        dpnl.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dlvh.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = dpnl.len() as f64;
        let pct_beat_pnl = dpnl.iter().filter(|&&d| d > 0.0).count() as f64 / n * 100.0;
        let pct_beat_lvh = dlvh.iter().filter(|&&d| d > 0.0).count() as f64 / n * 100.0;

        writeln!(
            sf2,
            "{},{:.4},{:.4},{:.4},{:.1},{:.4},{:.4},{:.1}",
            policy,
            mean(&dpnl),
            percentile(&dpnl, 50.0),
            percentile(&dpnl, 5.0),
            pct_beat_pnl,
            mean(&dlvh),
            percentile(&dlvh, 50.0),
            pct_beat_lvh,
        )
        .unwrap();
    }
    eprintln!("written: {summary_path}");
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let _n = xs.len() as f64;
    let mx = mean(xs);
    let my = mean(ys);
    let num: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let dx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>().sqrt();
    let dy: f64 = ys.iter().map(|y| (y - my).powi(2)).sum::<f64>().sqrt();
    if dx == 0.0 || dy == 0.0 {
        return 0.0;
    }
    num / (dx * dy)
}
