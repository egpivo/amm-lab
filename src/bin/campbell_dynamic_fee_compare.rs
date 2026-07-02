use amm_lab::campbell::fee_policy::{FixedFeePolicy, InventoryGapFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{SimConfig, run_simulation};
use std::env;
use std::fs;
use std::io::Write;

fn main() {
    let toml_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "scenarios/campbell_sim.toml".to_string());
    let toml_str =
        fs::read_to_string(&toml_path).unwrap_or_else(|e| panic!("cannot read {toml_path}: {e}"));
    let config: SimConfig =
        toml::from_str(&toml_str).unwrap_or_else(|e| panic!("invalid TOML: {e}"));

    let n_paths: usize = 500;
    fs::create_dir_all("data/processed").unwrap();
    let csv_path = "data/processed/campbell_sim_compare.csv";
    let mut f = fs::File::create(csv_path).unwrap();
    writeln!(
        f,
        "policy,seed,avg_fee_bps,hedged_pnl,lp_vs_hold,fee_revenue,lvr"
    )
    .unwrap();
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

    for (policy_name, make_policy) in policies {
        for path in 0..n_paths {
            let seed = config.seed + path as u64;
            let cex_prices = generate_gbm(
                config.n_steps,
                2000.0,
                config.mu,
                config.sigma,
                1.0 / config.n_steps as f64,
                seed,
            );
            let mut policy = make_policy();
            let records = run_simulation(&config, &cex_prices, &mut *policy);
            let total_fee: f64 = records.iter().map(|r| r.step_fee).sum();
            let last = records.last().unwrap();
            let lvr = last.hedging_portfolio - last.pool_value;
            let hedged_pnl = total_fee - lvr;
            let hold_value = config.reserve_x + config.reserve_y * last.cex_price;
            let lp_vs_hold = (last.pool_value + total_fee) - hold_value;
            let avg_fee_bps =
                records.iter().map(|r| r.fee_used * 10_000.0).sum::<f64>() / records.len() as f64;
            writeln!(
                f,
                "{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                policy_name, seed, avg_fee_bps, hedged_pnl, lp_vs_hold, total_fee, lvr
            )
            .unwrap();
        }
    }
    eprintln!("written: {csv_path}");
}
