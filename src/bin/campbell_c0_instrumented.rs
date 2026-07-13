//! Pre-C1 instrumentation run (C0 regression + extended per-path columns).
//!
//! Additive binary: does NOT modify existing outputs. Re-evaluates the five C0 policies
//! on the held-out CRN seeds and writes an extended per-path CSV (flow split by trade
//! leg, fee distribution stats, fee-revenue split). No policy changes, no training.
//!
//! Usage: campbell_c0_instrumented [scenario.toml] [rl_table.csv]

use amm_lab::campbell::fee_policy::{
    FixedFeePolicy, InventoryGapFeePolicy, OracleGapFeePolicy, TabularLearnedFeePolicy,
};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{SimConfig, StepRecord, run_simulation};
use std::io::Write;

const EVAL_START: u64 = 5_000;
const EVAL_PATHS: u64 = 500;
const INITIAL_PRICE: f64 = 1.0;

struct Ext {
    avg_fee_bps: f64,
    fee_std_bps: f64,
    fee_min_bps: f64,
    fee_max_bps: f64,
    hedged_pnl: f64,
    fee_revenue: f64,
    fee_revenue_arb: f64,
    fee_revenue_fund: f64,
    lvr: f64,
    fundamental_volume: f64,
    arb_volume: f64,
    volume: f64,
    fundamental_count: usize,
    arb_count: usize,
    avg_fundamental_size: f64,
    avg_arb_size: f64,
    final_external_price: f64,
}

fn summarize(records: &[StepRecord]) -> Ext {
    let n = records.len() as f64;
    let fee_revenue: f64 = records.iter().map(|r| r.step_fee).sum();
    let fee_revenue_arb: f64 = records.iter().map(|r| r.step_fee_arb).sum();
    let fee_revenue_fund: f64 = records.iter().map(|r| r.step_fee_fund).sum();
    let last = records.last().unwrap();
    let lvr = last.hedging_portfolio - last.pool_value;

    let fees: Vec<f64> = records.iter().map(|r| r.fee_used * 10_000.0).collect();
    let mean_fee = fees.iter().sum::<f64>() / n;
    let fee_std = (fees.iter().map(|f| (f - mean_fee).powi(2)).sum::<f64>() / n).sqrt();
    let fee_min = fees.iter().cloned().fold(f64::INFINITY, f64::min);
    let fee_max = fees.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let fund_volume: f64 = records
        .iter()
        .map(|r| r.buy_delta.abs() + r.sell_delta.abs())
        .sum();
    let arb_volume: f64 = records.iter().map(|r| r.arb_delta.abs()).sum();
    let arb_count = records.iter().filter(|r| r.arb_delta.abs() > 1e-12).count();
    let fundamental_count = records
        .iter()
        .filter(|r| r.buy_delta.abs() + r.sell_delta.abs() > 1e-12)
        .count();

    Ext {
        avg_fee_bps: mean_fee,
        fee_std_bps: fee_std,
        fee_min_bps: fee_min,
        fee_max_bps: fee_max,
        hedged_pnl: fee_revenue - lvr,
        fee_revenue,
        fee_revenue_arb,
        fee_revenue_fund,
        lvr,
        fundamental_volume: fund_volume,
        arb_volume,
        volume: fund_volume + arb_volume,
        fundamental_count,
        arb_count,
        avg_fundamental_size: if fundamental_count > 0 {
            fund_volume / fundamental_count as f64
        } else {
            0.0
        },
        avg_arb_size: if arb_count > 0 {
            arb_volume / arb_count as f64
        } else {
            0.0
        },
        final_external_price: last.cex_price,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let toml_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("scenarios/campbell_rl_normal.toml");
    let rl_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("data/processed/campbell_rl_fee_table.csv");

    let toml_str = std::fs::read_to_string(toml_path).expect("read scenario toml");
    let base_config: SimConfig = toml::from_str(&toml_str).expect("parse scenario toml");
    println!("scenario: {} | rl table: {rl_path}", base_config.name);

    let mut tabular_rl = TabularLearnedFeePolicy::from_csv(rl_path).expect("load RL table");
    tabular_rl.set_inference();
    let dt = 1.0 / base_config.n_steps as f64;

    type PolicyFn = fn() -> Box<dyn amm_lab::campbell::fee_policy::FeePolicy>;
    let fixed_policies: &[(&str, PolicyFn)] = &[
        ("fixed_6bps", || Box::new(FixedFeePolicy::new(0.0006))),
        ("fixed_10bps", || Box::new(FixedFeePolicy::new(0.0010))),
        ("oracle_gap", || {
            Box::new(OracleGapFeePolicy {
                base_fee: 0.0006,
                gap_multiplier: 0.1,
                min_fee: 0.0001,
                max_fee: 0.0020,
            })
        }),
        ("inventory_gap", || {
            Box::new(InventoryGapFeePolicy {
                base_fee: 0.0006,
                gap_multiplier: 0.01,
                min_fee: 0.0001,
                max_fee: 0.0020,
            })
        }),
    ];

    let out_path = "data/processed/campbell_c0_instrumented.csv";
    let mut f = std::fs::File::create(out_path).unwrap();
    writeln!(
        f,
        "scenario,policy,seed,avg_fee_bps,fee_std_bps,fee_min_bps,fee_max_bps,\
                 hedged_pnl,fee_revenue,fee_revenue_arb,fee_revenue_fund,lvr,\
                 fundamental_volume,arb_volume,volume,fundamental_count,arb_count,\
                 avg_fundamental_size,avg_arb_size,final_external_price"
    )
    .unwrap();

    let write_row = |name: &str, seed: u64, m: &Ext, fh: &mut std::fs::File| {
        writeln!(fh, "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{:.6},{:.6},{:.6}",
            base_config.name, name, seed,
            m.avg_fee_bps, m.fee_std_bps, m.fee_min_bps, m.fee_max_bps,
            m.hedged_pnl, m.fee_revenue, m.fee_revenue_arb, m.fee_revenue_fund, m.lvr,
            m.fundamental_volume, m.arb_volume, m.volume,
            m.fundamental_count, m.arb_count,
            m.avg_fundamental_size, m.avg_arb_size, m.final_external_price).unwrap();
    };

    for (name, make_policy) in fixed_policies {
        for seed in EVAL_START..EVAL_START + EVAL_PATHS {
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
            let mut policy = make_policy();
            let records = run_simulation(&config, &cex, &mut *policy);
            write_row(name, seed, &summarize(&records), &mut f);
        }
        println!("done: {name}");
    }
    for seed in EVAL_START..EVAL_START + EVAL_PATHS {
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
        let records = run_simulation(&config, &cex, &mut tabular_rl);
        write_row("tabular_rl", seed, &summarize(&records), &mut f);
    }
    println!("done: tabular_rl\nsaved -> {out_path}");
}
