//! Quick wall-time benchmark: one 1-second-clock week episode per
//! stratum (5bp high-activity, 30bp moderate), static policy, hazards
//! from the frozen 54-cell manifest (medium arb, sigma 0.64, z mid).
//! Informs validation-grid compute budgeting only; not a paper artifact.

use amm_lab::campbell::fee_policy::FixedFeePolicy;
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{
    ArrivalModel, FlowRegime, SimConfig, run_simulation_with_events,
};
use std::time::Instant;

fn config(fee: f64, lam_arb: f64, lam_fund: f64) -> SimConfig {
    let y0 = 1.0e4;
    let d_ref = y0 * (1.0 - (1.0f64 + 0.01).powf(-0.5));
    SimConfig {
        name: "bench".into(),
        description: "episode benchmark".into(),
        amm_fee: fee,
        cex_fee: 0.0010,
        buy_demand: 0.0055 * d_ref,
        sell_demand: 0.0055 * d_ref,
        reserve_x: 2.0e7,
        reserve_y: y0,
        sigma: 0.64,
        mu: 0.0,
        n_steps: 604_800,
        seed: 999,
        flow_regime: FlowRegime::Normal,
        toxic_burst_prob: 0.0,
        toxic_burst_arb_scale: 1.0,
        toxic_burst_fund_scale: 1.0,
        regime_switch_period: 0,
        e1_lambda: 0.0,
        e1_fee_ref: 0.0006,
        e5_arb_prob: 1.0,
        policy_lag: 300,
        dt_hours: 1.0 / 3600.0,
        pooled_fund_arrival_rate_per_hour: Some(lam_fund),
        buy_arrival_share: 0.5,
        arb_arrival_rate_per_hour: Some(lam_arb),
        lookback_hours: 20.0,
        arrival_model: ArrivalModel::Poisson,
        log_inactive_arb: false,
    }
}

fn main() {
    // hazards from calibration_54_manifest.json, sigma=0.64 z=0.0055 medium
    for (name, fee, lam_arb, lam_fund) in
        [("5bp", 0.0005, 40.0, 384.0), ("30bp", 0.0030, 1.5, 48.0)]
    {
        let cfg = config(fee, lam_arb, lam_fund);
        let prices = generate_gbm(
            cfg.n_steps,
            2000.0,
            0.0,
            cfg.sigma,
            cfg.dt_years(),
            cfg.seed,
        );
        let t = Instant::now();
        let mut pol = FixedFeePolicy::new(fee);
        let (records, events) = run_simulation_with_events(&cfg, &prices, &mut pol);
        println!(
            "{name}: {:.2}s  steps={} events={} fills={}",
            t.elapsed().as_secs_f64(),
            records.len(),
            events.len(),
            events.iter().filter(|e| e.delta != 0.0).count()
        );
    }
}
