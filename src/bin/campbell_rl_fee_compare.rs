use amm_lab::campbell::fee_policy::{
    FixedFeePolicy, InventoryGapFeePolicy, OracleGapFeePolicy, TabularLearnedFeePolicy,
};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{SimConfig, StepRecord, run_simulation};
use std::collections::HashMap;
use std::io::Write;

struct PathMetrics {
    avg_fee_bps: f64,
    hedged_pnl: f64,
    lp_vs_hold: f64,
    lp_vs_hold_pct: f64,
    fee_revenue: f64,
    lvr: f64,
    volume: f64,
    arb_count: usize,
    fundamental_count: usize,
    final_external_price: f64,
    final_pool_price: f64,
}

fn summarize(records: &[StepRecord], config: &SimConfig, initial_price: f64) -> PathMetrics {
    let fee_revenue: f64 = records.iter().map(|r| r.step_fee).sum();
    let last = records.last().unwrap();
    let lvr = last.hedging_portfolio - last.pool_value;
    let hedged_pnl = fee_revenue - lvr;

    let avg_fee_bps =
        records.iter().map(|r| r.fee_used * 10_000.0).sum::<f64>() / records.len() as f64;

    let final_price = last.cex_price;
    let initial_hold = config.reserve_x + config.reserve_y * initial_price;
    let final_hold = config.reserve_x + config.reserve_y * final_price;
    let lp_vs_hold = last.pool_value - final_hold;
    let lp_vs_hold_pct = lp_vs_hold / initial_hold * 100.0;

    let volume: f64 = records
        .iter()
        .map(|r| r.arb_delta.abs() + r.buy_delta.abs() + r.sell_delta.abs())
        .sum();
    let arb_count = records.iter().filter(|r| r.arb_delta.abs() > 1e-12).count();
    let fund_count = records
        .iter()
        .filter(|r| r.buy_delta.abs() + r.sell_delta.abs() > 1e-12)
        .count();

    PathMetrics {
        avg_fee_bps,
        hedged_pnl,
        lp_vs_hold,
        lp_vs_hold_pct,
        fee_revenue,
        lvr,
        volume,
        arb_count,
        fundamental_count: fund_count,
        final_external_price: final_price,
        final_pool_price: last.amm_price,
    }
}

fn main() {
    let config = SimConfig {
        name: "rl_eval".to_string(),
        description: "RL evaluation".to_string(),
        amm_fee: 0.0006,
        cex_fee: 0.0010,
        buy_demand: 100.0,
        sell_demand: 100.0,
        reserve_x: 1000.0,
        reserve_y: 1000.0,
        sigma: 0.04,
        mu: 0.0,
        n_steps: 1440,
        seed: 0,
    };
    let dt = 1.0 / config.n_steps as f64;
    const EVAL_START: u64 = 5000;
    const EVAL_PATHS: u64 = 500;
    const INITIAL_PRICE: f64 = 1.0;

    let mut tabular_rl =
        TabularLearnedFeePolicy::from_csv("data/processed/campbell_rl_fee_table.csv")
            .expect("load Q-table");
    tabular_rl.set_inference();

    // policy factory: (name, closure that creates fresh policy each seed)
    type PolicyFn = fn() -> Box<dyn amm_lab::campbell::fee_policy::FeePolicy>;
    let policies: &[(&str, PolicyFn)] = &[
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

    // collect results: (policy_name, seed) -> PathMetrics
    let mut results: Vec<(String, u64, PathMetrics)> = Vec::new();

    // run fixed policies
    for (name, make_policy) in policies {
        for seed in EVAL_START..EVAL_START + EVAL_PATHS {
            let cex_prices = generate_gbm(
                config.n_steps,
                INITIAL_PRICE,
                config.mu,
                config.sigma,
                dt,
                seed,
            );
            let mut policy = make_policy();
            let records = run_simulation(&config, &cex_prices, &mut *policy);
            let m = summarize(&records, &config, INITIAL_PRICE);
            results.push((name.to_string(), seed, m));
        }
        println!("done: {}", name);
    }

    // run tabular RL separately (stateless in inference)
    for seed in EVAL_START..EVAL_START + EVAL_PATHS {
        let cex_prices = generate_gbm(
            config.n_steps,
            INITIAL_PRICE,
            config.mu,
            config.sigma,
            dt,
            seed,
        );
        let records = run_simulation(&config, &cex_prices, &mut tabular_rl);
        let m = summarize(&records, &config, INITIAL_PRICE);
        results.push(("tabular_rl".to_string(), seed, m));
    }
    println!("done tabular_rl");

    // write compare csv
    std::fs::create_dir_all("data/processed").unwrap();
    {
        let mut f = std::fs::File::create("data/processed/campbell_rl_fee_compare.csv").unwrap();
        writeln!(
            f,
            "policy,seed,avg_fee_bps,hedged_pnl,lp_vs_hold,lp_vs_hold_pct,\
        fee_revenues,lvr,volume,arb_count,fundamental_count,\
        final_external_price,final_pool_price"
        )
        .unwrap();
        for (name, seed, m) in &results {
            writeln!(
                f,
                "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{:.6},{:.6}",
                name,
                seed,
                m.avg_fee_bps,
                m.hedged_pnl,
                m.lp_vs_hold,
                m.lp_vs_hold_pct,
                m.fee_revenue,
                m.lvr,
                m.volume,
                m.arb_count,
                m.fundamental_count,
                m.final_external_price,
                m.final_pool_price
            )
            .unwrap();
        }
    }
    println!("Saved -> data/processed/campbell_rl_fee_compare.csv");

    // build oracle_gap results by seed for paired delta
    let oracle_by_seed: HashMap<u64, &PathMetrics> = results
        .iter()
        .filter(|(name, _, _)| name == "oracle_gap")
        .map(|(_, seed, m)| (*seed, m))
        .collect();

    // write paired delta CSV
    {
        let mut f =
            std::fs::File::create("data/processed/campbell_rl_paired_delta_vs_oracle_gap.csv")
                .unwrap();
        writeln!(
            f,
            "policy,seed,delta_hedged_pnl_vs_oracle_gap,\
                     delta_lp_vs_hold_vs_oracle_gap,delta_fee_revenue_vs_oracle_gap,\
                     delta_lvr_vs_oracle_gap,delta_volume_vs_oracle_gap"
        )
        .unwrap();
        for (name, seed, m) in &results {
            if name == "oracle_gap" {
                continue;
            }
            if let Some(og) = oracle_by_seed.get(seed) {
                writeln!(
                    f,
                    "{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    name,
                    seed,
                    m.hedged_pnl - og.hedged_pnl,
                    m.lp_vs_hold - og.lp_vs_hold,
                    m.fee_revenue - og.fee_revenue,
                    m.lvr - og.lvr,
                    m.volume - og.volume
                )
                .unwrap();
            }
        }
    }
    println!("Saved → data/processed/campbell_rl_paired_delta_vs_oracle_gap.csv");

    // quick summary
    for policy_name in &[
        "fixed_6bps",
        "fixed_10bps",
        "oracle_gap",
        "inventory_gap",
        "tabular_rl",
    ] {
        let pnls: Vec<f64> = results
            .iter()
            .filter(|(n, _, _)| n == policy_name)
            .map(|(_, _, m)| m.hedged_pnl)
            .collect();
        let mean = pnls.iter().sum::<f64>() / pnls.len() as f64;
        let fees: Vec<f64> = results
            .iter()
            .filter(|(n, _, _)| n == policy_name)
            .map(|(_, _, m)| m.avg_fee_bps)
            .collect();
        let mean_fee = fees.iter().sum::<f64>() / fees.len() as f64;
        println!(
            "{}: mean hedged_pnl={:.4}, mean fee={:.2} bps",
            policy_name, mean, mean_fee
        );
    }
}
