use amm_lab::campbell::fee_policy::FixedFeePolicy;
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{SimConfig, SimSummary, run_simulation};
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
    let cex_prices = generate_gbm(
        config.n_steps,
        2000.0,
        config.mu,
        config.sigma,
        1.0 / config.n_steps as f64,
        config.seed,
    );
    let mut mul_policy = FixedFeePolicy::new(config.amm_fee);
    let records = run_simulation(&config, &cex_prices, &mut mul_policy);

    let total_fee: f64 = records.iter().map(|r| r.step_fee).sum();
    let last = records.last().unwrap();
    let tracking_error = last.hedging_portfolio - last.pool_value;
    let hedged_pnl = total_fee - tracking_error;

    let summary = SimSummary {
        scenario_name: "campbell_sim".to_string(),
        config: config.clone(),
        n_steps: records.len(),
        initial_cex_price: cex_prices[0],
        final_cex_price: last.cex_price,
        final_amm_price: last.amm_price,
        final_pool_value: last.pool_value,
        final_hedging_portfolio: last.hedging_portfolio,
        total_fee_revenue: total_fee,
        tracking_error,
        hedged_pnl,
    };

    fs::create_dir_all("data/processed").expect("could not create data/processed");

    // JSON summary
    let json_path = "data/processed/campbell_sim.json";
    let json = serde_json::to_string_pretty(&summary).expect("json serialization failed");
    fs::write(json_path, &json).expect("could not write JSON");

    // CSV steps
    let csv_path = "data/processed/campbell_sim_steps.csv";
    let mut f = fs::File::create(csv_path).expect("could not create CSV");
    writeln!(f, "step,cex_price,amm_price,arb_delta,buy_delta,sell_delta,step_fee,pool_value,hedging_portfolio")
        .unwrap();
    for r in &records {
        writeln!(
            f,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4}",
            r.step,
            r.cex_price,
            r.amm_price,
            r.arb_delta,
            r.buy_delta,
            r.sell_delta,
            r.step_fee,
            r.pool_value,
            r.hedging_portfolio,
        )
        .unwrap();
    }

    eprintln!("written: {json_path}");
    eprintln!("written: {csv_path}");
    eprintln!("--- Summary ---");
    eprintln!("Total fee revenue: {:.2}", total_fee);
    eprintln!("Tracking error:    {:.2}", tracking_error);
    eprintln!("Hedged PnL:        {:.2}", hedged_pnl);
}
