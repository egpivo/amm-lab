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
    let step_csv_path = "data/processed/campbell_dynamic_fee_steps.csv";
    let mut sf = fs::File::create(step_csv_path).unwrap();
    writeln!(sf, "policy,step,external_price,amm_price,oracle_gap_bps,inventory_skew,recent_vol,fee_bps,trade_type,pool_x,pool_y,fee_revenue,lvr").unwrap();

    let sample_prices = generate_gbm(
        config.n_steps,
        2000.0,
        config.mu,
        config.sigma,
        1.0 / config.n_steps as f64,
        config.seed,
    );

    for (policy_name, make_policy) in policies {
        let mut policy = make_policy();
        let records = run_simulation(&config, &sample_prices, &mut *policy);
        for r in &records {
            let vol = compute_recent_vol(&sample_prices, r.step, 20);
            let trade_type = match (
                r.arb_delta != 0.0,
                r.buy_delta != 0.0 || r.sell_delta != 0.0,
            ) {
                (true, true) => "arb+fund",
                (true, false) => "arb",
                (false, true) => "fund",
                (false, false) => "none",
            };
            let lvr = r.hedging_portfolio - r.pool_value;
            writeln!(
                sf,
                "{},{},{:.6},{:.6},{:.4},{:.6},{:.8},{:.4},{},{:.4},{:.4},{:.6},{:.4}",
                policy_name,
                r.step,
                r.cex_price,
                r.amm_price,
                r.oracle_gap_bps,
                r.inventory_skew,
                vol,
                r.fee_used * 10_000.0,
                trade_type,
                r.pool_x,
                r.pool_y,
                r.step_fee,
                lvr
            )
            .unwrap();
        }
    }
    eprintln!("written: {csv_path}, {step_csv_path}");
}

fn compute_recent_vol(cex_prices: &[f64], step: usize, window: usize) -> f64 {
    let lo = (step + 1).saturating_sub(window);
    let slice = &cex_prices[lo..=step + 1];
    if slice.len() < 2 {
        return 0.0;
    }
    let n = (slice.len() - 1) as f64;
    let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
    for w in slice.windows(2) {
        let lr = (w[1] / w[0]).ln();
        sum += lr;
        sum_sq += lr * lr;
    }
    let mean = sum / n;
    ((sum_sq / n - mean * mean).max(0.0)).sqrt()
}
