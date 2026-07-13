//! C1a / E1 router-substitution stress (spec: .local/amm_causal_c1a_e1/C1A_E1_SPEC.md).
//!
//! CONTROLLED SIMULATION — PAIRED CAUSAL CONTRAST — PARTIAL EQUILIBRIUM + E1 SUBSTITUTION
//! STRESS. Four policies x four pre-registered lambda levels x 500 CRN seeds. Fundamental
//! flow only; arbitrage unchanged. tabular_rl excluded (schema break, disclosed).

use amm_lab::campbell::fee_policy::{FixedFeePolicy, InventoryGapFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{SimConfig, StepRecord, run_simulation};
use std::io::Write;

const EVAL_START: u64 = 5_000;
const EVAL_PATHS: u64 = 500;
const INITIAL_PRICE: f64 = 1.0;
const LAMBDAS: [(&str, f64); 4] = [
    ("E1_none", 0.0),
    ("E1_low", 0.25),
    ("E1_mid", 0.5),
    ("E1_high", 1.0),
];

fn main() {
    let toml_str = std::fs::read_to_string("scenarios/campbell_rl_normal.toml").unwrap();
    let base: SimConfig = toml::from_str(&toml_str).unwrap();
    let dt = 1.0 / base.n_steps as f64;

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

    let out = "data/processed/campbell_c1a_e1.csv";
    let mut f = std::fs::File::create(out).unwrap();
    writeln!(
        f,
        "elasticity,lambda,policy,seed,avg_fee_bps,fee_std_bps,fee_min_bps,fee_max_bps,\
                 hedged_pnl,fee_revenue,fee_revenue_arb,fee_revenue_fund,lvr,\
                 fundamental_volume,arb_volume,volume,fundamental_count,arb_count,\
                 avg_fundamental_size,avg_arb_size,mean_retention,fund_demand_lost,\
                 final_external_price"
    )
    .unwrap();

    for (lname, lam) in LAMBDAS {
        for (pname, make_policy) in policies {
            for seed in EVAL_START..EVAL_START + EVAL_PATHS {
                let mut config = base.clone();
                config.seed = seed;
                config.e1_lambda = lam;
                config.e1_fee_ref = 0.0006;
                let cex = generate_gbm(
                    config.n_steps,
                    INITIAL_PRICE,
                    config.mu,
                    config.sigma,
                    dt,
                    seed,
                );
                let mut policy = make_policy();
                let r: Vec<StepRecord> = run_simulation(&config, &cex, &mut *policy);

                let n = r.len() as f64;
                let fee_revenue: f64 = r.iter().map(|x| x.step_fee).sum();
                let fee_arb: f64 = r.iter().map(|x| x.step_fee_arb).sum();
                let fee_fund: f64 = r.iter().map(|x| x.step_fee_fund).sum();
                let last = r.last().unwrap();
                let lvr = last.hedging_portfolio - last.pool_value;
                let fees: Vec<f64> = r.iter().map(|x| x.fee_used * 10_000.0).collect();
                let mfee = fees.iter().sum::<f64>() / n;
                let fstd = (fees.iter().map(|v| (v - mfee).powi(2)).sum::<f64>() / n).sqrt();
                let fmin = fees.iter().cloned().fold(f64::INFINITY, f64::min);
                let fmax = fees.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let fv: f64 = r
                    .iter()
                    .map(|x| x.buy_delta.abs() + x.sell_delta.abs())
                    .sum();
                let av: f64 = r.iter().map(|x| x.arb_delta.abs()).sum();
                let fc = r
                    .iter()
                    .filter(|x| x.buy_delta.abs() + x.sell_delta.abs() > 1e-12)
                    .count();
                let ac = r.iter().filter(|x| x.arb_delta.abs() > 1e-12).count();
                let mret: f64 = r.iter().map(|x| x.fund_retention).sum::<f64>() / n;
                let dlost: f64 = r.iter().map(|x| x.fund_demand_lost).sum();

                writeln!(
                    f,
                    "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},\
                             {:.4},{:.4},{:.4},{},{},{:.6},{:.6},{:.6},{:.4},{:.6}",
                    lname,
                    lam,
                    pname,
                    seed,
                    mfee,
                    fstd,
                    fmin,
                    fmax,
                    fee_revenue - lvr,
                    fee_revenue,
                    fee_arb,
                    fee_fund,
                    lvr,
                    fv,
                    av,
                    fv + av,
                    fc,
                    ac,
                    if fc > 0 { fv / fc as f64 } else { 0.0 },
                    if ac > 0 { av / ac as f64 } else { 0.0 },
                    mret,
                    dlost,
                    last.cex_price
                )
                .unwrap();
            }
        }
        println!("done: {lname}");
    }
    println!("saved -> {out}");
}
