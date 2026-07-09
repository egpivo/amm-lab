//! E5 arbitrage-latency stress (spec: .local/amm_causal_e5_latency/E5_SPEC.md).
//!
//! CONTROLLED SIMULATION — E5 LATENCY STRESS — PARTIAL EQUILIBRIUM (ROUTING FROZEN).
//! Four policies x pre-registered q in {1.0, 0.5, 0.25, 0.125} x 500 CRN seeds.
//! Latency draws come from a dedicated exogenous stream (seed ^ 77_777), identical
//! across policies and q-independent in alignment. tabular_rl excluded (schema break).

use amm_lab::campbell::fee_policy::{FixedFeePolicy, InventoryGapFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{SimConfig, run_simulation};
use std::io::Write;

const EVAL_START: u64 = 5_000;
const EVAL_PATHS: u64 = 500;
const INITIAL_PRICE: f64 = 1.0;
const LEVELS: [(&str, f64); 4] = [
    ("E5_none", 1.0),
    ("E5_low", 0.5),
    ("E5_mid", 0.25),
    ("E5_high", 0.125),
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

    let out = "data/processed/campbell_e5_latency.csv";
    let mut f = std::fs::File::create(out).unwrap();
    writeln!(
        f,
        "latency,arb_prob,policy,seed,avg_fee_bps,fee_std_bps,fee_max_bps,\
                 hedged_pnl,fee_revenue,fee_revenue_arb,fee_revenue_fund,lvr,\
                 fundamental_volume,arb_volume,volume,fundamental_count,arb_count,\
                 arb_opportunities,mean_abs_gap_bps,final_external_price"
    )
    .unwrap();

    for (lname, q) in LEVELS {
        for (pname, make_policy) in policies {
            for seed in EVAL_START..EVAL_START + EVAL_PATHS {
                let mut config = base.clone();
                config.seed = seed;
                config.e5_arb_prob = q;
                config.e1_lambda = 0.0; // routing frozen (spec)
                let cex = generate_gbm(
                    config.n_steps,
                    INITIAL_PRICE,
                    config.mu,
                    config.sigma,
                    dt,
                    seed,
                );
                let mut policy = make_policy();
                let r = run_simulation(&config, &cex, &mut *policy);

                let n = r.len() as f64;
                let fee_revenue: f64 = r.iter().map(|x| x.step_fee).sum();
                let fee_arb: f64 = r.iter().map(|x| x.step_fee_arb).sum();
                let fee_fund: f64 = r.iter().map(|x| x.step_fee_fund).sum();
                let last = r.last().unwrap();
                let lvr = last.hedging_portfolio - last.pool_value;
                let fees: Vec<f64> = r.iter().map(|x| x.fee_used * 10_000.0).collect();
                let mfee = fees.iter().sum::<f64>() / n;
                let fstd = (fees.iter().map(|v| (v - mfee).powi(2)).sum::<f64>() / n).sqrt();
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
                let opp = r.iter().filter(|x| x.arb_active).count();
                let mgap: f64 = r.iter().map(|x| x.oracle_gap_bps.abs()).sum::<f64>() / n;

                writeln!(
                    f,
                    "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},\
                             {:.4},{:.4},{:.4},{},{},{},{:.4},{:.6}",
                    lname,
                    q,
                    pname,
                    seed,
                    mfee,
                    fstd,
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
                    opp,
                    mgap,
                    last.cex_price
                )
                .unwrap();
            }
        }
        println!("done: {lname}");
    }
    println!("saved -> {out}");
}
