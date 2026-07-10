//! M2 artifact diagnostics for the trained DynamicDuopoly learner.
//!
//! The Q-table trained under the base DynamicDuopoly config is FROZEN and
//! evaluated under perturbed environments (no retraining), against the
//! strongest tuned baseline (lookahead) and TWAP. Diagnostics:
//!   priority   : agent order Before / After / Random vs noise+arb
//!   gas        : agent gas 0 / 2 / 10 X
//!   arb_speed  : 0.2 / 0.5 / 1.0
//!   noise      : shifted flow distribution (intensity x2, sigma x2)
//!   coeff      : a_own / a_rival / a_oracle scaled x0.5 / x2
//!   mode       : transfer to ConstantDuopoly / DynamicMonopoly
//!   fresh_test : base config on seeds never touched during M1 iteration

use amm_lab::sim::env::{AgentOrder, EnvConfig, EpisodeSummary, ExecEnv, MarketMode};
use amm_lab::sim::execution_agent::{ExecutionPolicy, LookaheadPolicy, TwapPolicy};
use amm_lab::sim::q_learner::{QPolicy, QTable};
use clap::Parser;
use std::io::Write as _;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 300)]
    n_seeds: u64,
    #[arg(long, default_value_t = 500)]
    n_fresh: u64,
    #[arg(long, default_value = "experiments/rl_execution/out")]
    out_dir: String,
    /// kappa selected on validation seeds in M1 (dynamic duopoly).
    #[arg(long, default_value_t = 16.0)]
    kappa: f64,
}

const TEST_BASE: u64 = 30_000;
const FRESH_BASE: u64 = 40_000;

fn run(cfg: EnvConfig, policy: &mut dyn ExecutionPolicy) -> EpisodeSummary {
    let mut env = ExecEnv::new(cfg);
    policy.reset();
    while !env.is_done() {
        let obs = env.observe();
        let a = policy.act(&obs);
        env.step(a);
    }
    env.summary(policy.name())
}

fn main() {
    let args = Args::parse();
    let horizon = EnvConfig::baseline(MarketMode::DynamicDuopoly, 0)
        .order
        .horizon;
    let table = QTable::load(&format!("{}/q_table_dynamic_duopoly.json", args.out_dir))
        .expect("q table (run rl_equilibrium_train_tabular first)");

    // (diagnostic, variant, config-mutator)
    type Mutator = Box<dyn Fn(&mut EnvConfig)>;
    let mut cells: Vec<(String, String, Mutator)> = Vec::new();
    let cell = |d: &str, v: &str, f: Mutator| (d.to_string(), v.to_string(), f);

    cells.push(cell("priority", "before", Box::new(|_| {})));
    cells.push(cell(
        "priority",
        "after",
        Box::new(|c| c.agent_order = AgentOrder::After),
    ));
    cells.push(cell(
        "priority",
        "random",
        Box::new(|c| c.agent_order = AgentOrder::Random),
    ));
    for gas in [0.0, 2.0, 10.0] {
        cells.push(cell(
            "gas",
            &format!("gas={gas}"),
            Box::new(move |c| c.agent_gas_cost = gas),
        ));
    }
    for speed in [0.2, 0.5, 1.0] {
        cells.push(cell(
            "arb_speed",
            &format!("speed={speed}"),
            Box::new(move |c| c.arb.speed = speed),
        ));
    }
    cells.push(cell("noise", "base", Box::new(|_| {})));
    cells.push(cell(
        "noise",
        "intensity_x2",
        Box::new(|c| {
            c.noise.buy_intensity *= 2.0;
            c.noise.sell_intensity *= 2.0;
        }),
    ));
    cells.push(cell(
        "noise",
        "sigma_x2",
        Box::new(|c| c.noise.volume_sigma *= 2.0),
    ));
    for (name, f) in [
        (
            "a_own_x0.5",
            Box::new(|c: &mut EnvConfig| c.dynamic_fee.a_own *= 0.5) as Mutator,
        ),
        (
            "a_own_x2",
            Box::new(|c: &mut EnvConfig| c.dynamic_fee.a_own *= 2.0),
        ),
        (
            "a_rival_x0.5",
            Box::new(|c: &mut EnvConfig| c.dynamic_fee.a_rival *= 0.5),
        ),
        (
            "a_rival_x2",
            Box::new(|c: &mut EnvConfig| c.dynamic_fee.a_rival *= 2.0),
        ),
        (
            "a_oracle_x0.5",
            Box::new(|c: &mut EnvConfig| c.dynamic_fee.a_oracle *= 0.5),
        ),
        (
            "a_oracle_x2",
            Box::new(|c: &mut EnvConfig| c.dynamic_fee.a_oracle *= 2.0),
        ),
    ] {
        cells.push(cell("coeff", name, f));
    }
    cells.push(cell(
        "mode",
        "constant_duopoly",
        Box::new(|c| c.mode = MarketMode::ConstantDuopoly),
    ));
    cells.push(cell(
        "mode",
        "dynamic_monopoly",
        Box::new(|c| c.mode = MarketMode::DynamicMonopoly),
    ));
    cells.push(cell("mode", "dynamic_duopoly", Box::new(|_| {})));

    let out_path = format!("{}/m2_diagnostics.csv", args.out_dir);
    let mut f = std::fs::File::create(&out_path).expect("create csv");
    writeln!(
        f,
        "diagnostic,variant,policy,seed,shortfall_bps,completion_rate,total_reward,gas_paid_bps,fee_paid_bps"
    )
    .unwrap();

    let run_cell = |diag: &str,
                    variant: &str,
                    mutate: &Mutator,
                    seeds: std::ops::Range<u64>,
                    f: &mut std::fs::File| {
        let mut means: Vec<(String, f64, f64)> = Vec::new();
        for policy_name in ["q_learner", "lookahead", "twap"] {
            let mut sum_is = 0.0;
            let mut sum_comp = 0.0;
            let mut n = 0.0;
            for seed in seeds.clone() {
                let mut cfg = EnvConfig::baseline(MarketMode::DynamicDuopoly, seed);
                mutate(&mut cfg);
                let mut policy: Box<dyn ExecutionPolicy> = match policy_name {
                    "q_learner" => Box::new(QPolicy {
                        table: table.clone(),
                        horizon,
                    }),
                    "lookahead" => Box::new(LookaheadPolicy {
                        horizon,
                        kappa: args.kappa,
                        unfinished_penalty: 0.02,
                    }),
                    _ => Box::new(TwapPolicy { horizon }),
                };
                let s = run(cfg, policy.as_mut());
                writeln!(
                    f,
                    "{diag},{variant},{},{},{:.4},{:.4},{:.6},{:.4},{:.4}",
                    s.policy,
                    s.seed,
                    s.shortfall_bps,
                    s.completion_rate,
                    s.total_reward,
                    s.gas_paid_bps,
                    s.fee_paid_bps
                )
                .unwrap();
                sum_is += s.shortfall_bps;
                sum_comp += s.completion_rate;
                n += 1.0;
            }
            means.push((policy_name.to_string(), sum_is / n, sum_comp / n));
        }
        let q = means.iter().find(|m| m.0 == "q_learner").unwrap();
        let la = means.iter().find(|m| m.0 == "lookahead").unwrap();
        println!(
            "{diag:<10} {variant:<18} q {:>8.2} (comp {:.3})   lookahead {:>8.2}   edge {:+.2} bps",
            q.1,
            q.2,
            la.1,
            q.1 - la.1
        );
    };

    println!(
        "--- perturbation cells (test seeds {TEST_BASE}..{}) ---",
        TEST_BASE + args.n_seeds
    );
    for (diag, variant, mutate) in &cells {
        run_cell(
            diag,
            variant,
            mutate,
            TEST_BASE..TEST_BASE + args.n_seeds,
            &mut f,
        );
    }
    println!("--- fresh seeds (never used in M1 iteration) ---");
    let ident: Mutator = Box::new(|_| {});
    run_cell(
        "fresh_test",
        "base",
        &ident,
        FRESH_BASE..FRESH_BASE + args.n_fresh,
        &mut f,
    );
    println!("wrote {out_path}");
}
