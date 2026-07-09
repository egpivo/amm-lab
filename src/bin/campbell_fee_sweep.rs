/// Fixed-fee sweep diagnostic.
/// Usage: cargo run --bin campbell_fee_sweep -- scenarios/campbell_rl_normal.toml
///
/// Runs fixed-fee policies at [1,3,6,10,30,50,100] bps + oracle_gap on 500 eval seeds.
/// Prints: hedged_pnl, p05, fee_bps, total/arb/fund volume, fee_revenue, LVR.
/// Purpose: verify whether LVR is fee-sensitive in this simulator.
use amm_lab::campbell::fee_policy::{FeePolicy, FixedFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{SimConfig, StepRecord, run_simulation};
use std::io::Write;

const EVAL_START: u64 = 5_000;
const EVAL_PATHS: u64 = 500;
const INITIAL_PRICE: f64 = 1.0;

const SWEEP_BPS: &[f64] = &[1.0, 3.0, 6.0, 10.0, 30.0, 50.0, 100.0];

struct Metrics {
    hedged_pnl: f64,
    fee_revenue: f64,
    lvr: f64,
    volume: f64,
    arb_volume: f64,
    fund_volume: f64,
    avg_fee_bps: f64,
}

fn summarize(records: &[StepRecord]) -> Metrics {
    let fee_revenue: f64 = records.iter().map(|r| r.step_fee).sum();
    let last = records.last().unwrap();
    let lvr = last.hedging_portfolio - last.pool_value;
    let hedged_pnl = fee_revenue - lvr;
    let volume: f64 = records
        .iter()
        .map(|r| r.arb_delta.abs() + r.buy_delta.abs() + r.sell_delta.abs())
        .sum();
    let arb_volume: f64 = records.iter().map(|r| r.arb_delta.abs()).sum();
    let fund_volume: f64 = records
        .iter()
        .map(|r| r.buy_delta.abs() + r.sell_delta.abs())
        .sum();
    let avg_fee_bps =
        records.iter().map(|r| r.fee_used * 10_000.0).sum::<f64>() / records.len() as f64;
    Metrics {
        hedged_pnl,
        fee_revenue,
        lvr,
        volume,
        arb_volume,
        fund_volume,
        avg_fee_bps,
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn p05(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (0.05 * (v.len().saturating_sub(1)) as f64).round() as usize;
    v[idx]
}

fn run_policy(
    base_config: &SimConfig,
    make: &mut dyn FnMut() -> Box<dyn FeePolicy>,
) -> Vec<Metrics> {
    let dt = 1.0 / base_config.n_steps as f64;
    (EVAL_START..EVAL_START + EVAL_PATHS)
        .map(|seed| {
            let mut config = base_config.clone();
            config.seed = seed;
            let cex = generate_gbm(
                config.n_steps,
                INITIAL_PRICE,
                config.mu,
                config.sigma,
                dt,
                seed,
            );
            let mut policy = make();
            let records = run_simulation(&config, &cex, &mut *policy);
            summarize(&records)
        })
        .collect()
}

fn print_row(label: &str, ms: &[Metrics]) {
    let mut pnls: Vec<f64> = ms.iter().map(|m| m.hedged_pnl).collect();
    println!(
        "{:<13} {:>9.4} {:>9.4} {:>8.2} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
        label,
        mean(&pnls),
        p05(&mut pnls),
        mean(&ms.iter().map(|m| m.avg_fee_bps).collect::<Vec<_>>()),
        mean(&ms.iter().map(|m| m.volume).collect::<Vec<_>>()),
        mean(&ms.iter().map(|m| m.arb_volume).collect::<Vec<_>>()),
        mean(&ms.iter().map(|m| m.fund_volume).collect::<Vec<_>>()),
        mean(&ms.iter().map(|m| m.fee_revenue).collect::<Vec<_>>()),
        mean(&ms.iter().map(|m| m.lvr).collect::<Vec<_>>()),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let toml_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("scenarios/campbell_rl_normal.toml");
    let toml_str = std::fs::read_to_string(toml_path)
        .unwrap_or_else(|e| panic!("cannot read {toml_path}: {e}"));
    let base_config: SimConfig =
        toml::from_str(&toml_str).unwrap_or_else(|e| panic!("invalid TOML: {e}"));

    println!(
        "Scenario: {}  ({} eval paths)",
        base_config.name, EVAL_PATHS
    );
    println!();

    let sep = "─".repeat(102);
    println!("{sep}");
    println!(
        "{:<13} {:>9} {:>9} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "policy",
        "mean_pnl",
        "p05_pnl",
        "fee_bps",
        "tot_vol",
        "arb_vol",
        "fund_vol",
        "fee_rev",
        "lvr"
    );
    println!("{sep}");

    let mut csv_rows: Vec<(String, Vec<Metrics>)> = Vec::new();

    for &bps in SWEEP_BPS {
        let fee = bps / 10_000.0;
        let ms = run_policy(&base_config, &mut || Box::new(FixedFeePolicy::new(fee)));
        let label = format!("fixed_{:.0}bps", bps);
        print_row(&label, &ms);
        csv_rows.push((label, ms));
    }

    println!("{sep}");
    let og_ms = run_policy(&base_config, &mut || {
        Box::new(OracleGapFeePolicy {
            base_fee: 0.0006,
            gap_multiplier: 0.1,
            min_fee: 0.0001,
            max_fee: 0.0020,
        })
    });
    print_row("oracle_gap", &og_ms);
    csv_rows.push(("oracle_gap".to_string(), og_ms));
    println!("{sep}");

    std::fs::create_dir_all("data/processed").unwrap();
    let csv_path = format!("data/processed/campbell_fee_sweep_{}.csv", base_config.name);
    let mut f = std::fs::File::create(&csv_path).unwrap();
    writeln!(
        f,
        "scenario,policy,seed,hedged_pnl,fee_revenue,lvr,volume,arb_volume,fund_volume,avg_fee_bps"
    )
    .unwrap();
    for (label, ms) in &csv_rows {
        for (i, m) in ms.iter().enumerate() {
            writeln!(
                f,
                "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4}",
                base_config.name,
                label,
                EVAL_START + i as u64,
                m.hedged_pnl,
                m.fee_revenue,
                m.lvr,
                m.volume,
                m.arb_volume,
                m.fund_volume,
                m.avg_fee_bps,
            )
            .unwrap();
        }
    }
    println!("\nSaved → {csv_path}");
}
