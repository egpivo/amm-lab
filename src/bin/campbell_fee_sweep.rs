use amm_lab::campbell::fee_policy::FixedFeePolicy;
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{SimConfig, run_simulation};
use std::env;
use std::fs;
use std::io::Write;

fn main() {
    let toml_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "scenarios/campbell_fee_sweep.toml".to_string());
    let toml_str =
        fs::read_to_string(&toml_path).unwrap_or_else(|e| panic!("cannot read {toml_path}: {e}"));
    let config: SimConfig =
        toml::from_str(&toml_str).unwrap_or_else(|e| panic!("invalid TOML: {e}"));

    let cex_prices = generate_gbm(
        config.n_steps,
        2000.0,
        config.mu,
        config.sigma,
        1.0 / config.n_steps as f64,
        config.seed,
    );
    fs::create_dir_all("data/processed").unwrap();
    let csv_path = "data/processed/campbell_fee_sweep.csv";
    let mut f = fs::File::create(csv_path).unwrap();
    writeln!(f, "fee_bps,amm_fee,total_fee,tracking_error,hedged_pnl").unwrap();

    for fee_bps in 1u32..=100 {
        let amm_fee = fee_bps as f64 / 10_000.0;
        let mut sweep_config = config.clone();
        sweep_config.amm_fee = amm_fee;
        let mut mul_policy = FixedFeePolicy::new(amm_fee);
        let records = run_simulation(&sweep_config, &cex_prices, &mut mul_policy);

        let total_fee: f64 = records.iter().map(|r| r.step_fee).sum();
        let last = records.last().unwrap();
        let tracking_error = last.hedging_portfolio - last.pool_value;
        let hedged_pnl = total_fee - tracking_error;

        writeln!(
            f,
            "{},{:.6},{:.4},{:.4},{:.4}",
            fee_bps, amm_fee, total_fee, tracking_error, hedged_pnl
        )
        .unwrap();
    }
    eprintln!("written: {csv_path}");
}
