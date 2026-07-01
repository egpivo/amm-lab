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
    let csv_path = "data/processed/campbell_monte_carlo.csv";
    let mut f = fs::File::create(csv_path).unwrap();
    writeln!(f, "fee_bps,amm_fee,avg_hedged_pnl,std_hedged_pnl").unwrap();

    for fee_bps in 1u32..=100 {
        let amm_fee = fee_bps as f64 / 10_000.0;
        let mut sweep_config = config.clone();
        sweep_config.amm_fee = amm_fee;

        let mut pnl_samples: Vec<f64> = Vec::with_capacity(n_paths);
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
            let records = run_simulation(&sweep_config, &cex_prices);
            let total_fee: f64 = records.iter().map(|r| r.step_fee).sum();
            let last = records.last().unwrap();
            let hedged_pnl = total_fee - (last.hedging_portfolio - last.pool_value);
            pnl_samples.push(hedged_pnl);
        }
        let n = pnl_samples.len() as f64;
        let avg = pnl_samples.iter().sum::<f64>() / n;
        let variance = pnl_samples.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        writeln!(f, "{},{:.6},{:.4},{:.4}", fee_bps, amm_fee, avg, std).unwrap();
    }
    eprintln!("written: {csv_path}");
}
