use amm_lab::campbell::fee_policy::{
    FixedFeePolicy, InventoryGapFeePolicy, OracleGapFeePolicy, TabularLearnedFeePolicy,
};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{
    DEFAULT_RL_SCENARIO, FlowRegime, SimConfig, StepRecord, load_sim_config, run_simulation,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::io::Write;

const EVAL_START: u64 = 5_000;
const EVAL_PATHS: u64 = 500;
const INITIAL_PRICE: f64 = 1.0;

// ── Metrics ───────────────────────────────────────────────────────────────────

struct PathMetrics {
    avg_fee_bps: f64,
    hedged_pnl: f64,
    lp_vs_hold: f64,
    fee_revenue: f64,
    lvr: f64,
    volume: f64,
    arb_count: usize,
    fundamental_count: usize,
    final_external_price: f64,
    final_pool_price: f64,
    // regime-level aggregates (populated for regime_switch / toxic_burst)
    normal_hedged_pnl: f64,
    toxic_hedged_pnl: f64,
    normal_fee_revenue: f64,
    toxic_fee_revenue: f64,
    normal_steps: usize,
    toxic_steps: usize,
}

// Per-step hedged PnL increment: step_fee_t − Δ(hedging − pool_value)_t
fn per_step_hedged_pnl(records: &[StepRecord]) -> Vec<f64> {
    let mut out = Vec::with_capacity(records.len());
    let mut prev_pos = 0.0f64; // initial: hedging = pool_value → position = 0
    for r in records {
        let cur_pos = r.hedging_portfolio - r.pool_value;
        out.push(r.step_fee - (cur_pos - prev_pos));
        prev_pos = cur_pos;
    }
    out
}

// Returns true per step if the step is in the "toxic/arb-heavy" regime.
fn step_is_toxic(config: &SimConfig, eval_seed: u64) -> Vec<bool> {
    match &config.flow_regime {
        FlowRegime::Normal => vec![false; config.n_steps],
        FlowRegime::RegimeSwitch => {
            let period = config.regime_switch_period.max(1);
            (0..config.n_steps)
                .map(|i| !(i / period).is_multiple_of(2))
                .collect()
        }
        FlowRegime::ToxicBurst => {
            // Replay the same RNG used in run_simulation for this seed
            let mut rng = StdRng::seed_from_u64(eval_seed.wrapping_add(99_991));
            (0..config.n_steps)
                .map(|_| rng.gen_range(0.0f64..1.0) < config.toxic_burst_prob)
                .collect()
        }
    }
}

fn summarize(records: &[StepRecord], config: &SimConfig, eval_seed: u64) -> PathMetrics {
    let fee_revenue: f64 = records.iter().map(|r| r.step_fee).sum();
    let last = records.last().unwrap();
    let lvr = last.hedging_portfolio - last.pool_value;
    let hedged_pnl = fee_revenue - lvr;

    let avg_fee_bps =
        records.iter().map(|r| r.fee_used * 10_000.0).sum::<f64>() / records.len() as f64;

    let _initial_hold = config.reserve_x + config.reserve_y * INITIAL_PRICE;
    let final_hold = config.reserve_x + config.reserve_y * last.cex_price;
    let lp_vs_hold = last.pool_value - final_hold;

    let volume: f64 = records
        .iter()
        .map(|r| r.arb_delta.abs() + r.buy_delta.abs() + r.sell_delta.abs())
        .sum();
    let arb_count = records.iter().filter(|r| r.arb_delta.abs() > 1e-12).count();
    let fundamental_count = records
        .iter()
        .filter(|r| r.buy_delta.abs() + r.sell_delta.abs() > 1e-12)
        .count();

    // Regime-specific breakdown
    let pnl_per_step = per_step_hedged_pnl(records);
    let toxic_mask = step_is_toxic(config, eval_seed);

    let (
        normal_hedged_pnl,
        toxic_hedged_pnl,
        normal_fee_revenue,
        toxic_fee_revenue,
        normal_steps,
        toxic_steps,
    ) = records
        .iter()
        .enumerate()
        .fold((0.0, 0.0, 0.0, 0.0, 0usize, 0usize), |mut acc, (i, r)| {
            if toxic_mask[i] {
                acc.1 += pnl_per_step[i];
                acc.3 += r.step_fee;
                acc.5 += 1;
            } else {
                acc.0 += pnl_per_step[i];
                acc.2 += r.step_fee;
                acc.4 += 1;
            }
            acc
        });

    PathMetrics {
        avg_fee_bps,
        hedged_pnl,
        lp_vs_hold,
        fee_revenue,
        lvr,
        volume,
        arb_count,
        fundamental_count,
        final_external_price: last.cex_price,
        final_pool_price: last.amm_price,
        normal_hedged_pnl,
        toxic_hedged_pnl,
        normal_fee_revenue,
        toxic_fee_revenue,
        normal_steps,
        toxic_steps,
    }
}

// ── Summary stats ─────────────────────────────────────────────────────────────

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn percentile(sorted: &mut [f64], p: f64) -> f64 {
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (p / 100.0 * (sorted.len().saturating_sub(1)) as f64).round() as usize;
    sorted[idx]
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let toml_path = args.get(1).map(|s| s.as_str());
    let rl_csv_path = args.get(2).cloned();

    let base_config = load_sim_config(toml_path);
    let config_path = toml_path.unwrap_or(DEFAULT_RL_SCENARIO);

    // Resolve RL policy path: explicit arg > scenario-named > canonical
    let rl_path = rl_csv_path.unwrap_or_else(|| {
        let named = format!(
            "data/processed/campbell_rl_fee_table_{}.csv",
            base_config.name
        );
        if std::path::Path::new(&named).exists() {
            named
        } else {
            "data/processed/campbell_rl_fee_table.csv".to_string()
        }
    });

    println!("Scenario:  {}", base_config.name);
    println!("Config:    {config_path}");
    println!("Regime:    {:?}", base_config.flow_regime);
    println!("RL policy: {rl_path}");
    println!("Eval seeds: {EVAL_START}..{}", EVAL_START + EVAL_PATHS);
    println!();

    let mut tabular_rl = TabularLearnedFeePolicy::from_csv(&rl_path)
        .unwrap_or_else(|e| panic!("load Q-table from {rl_path}: {e}"));
    tabular_rl.set_inference();

    let dt = 1.0 / base_config.n_steps as f64;

    // policy registry: name + factory
    type PolicyFn = fn() -> Box<dyn amm_lab::campbell::fee_policy::FeePolicy>;
    let fixed_policies: &[(&str, PolicyFn)] = &[
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

    // results: (policy, seed) → PathMetrics
    let mut results: Vec<(String, u64, PathMetrics)> = Vec::new();

    for (name, make_policy) in fixed_policies {
        for seed in EVAL_START..EVAL_START + EVAL_PATHS {
            let mut config = base_config.clone();
            config.seed = seed; // vary regime RNG per episode
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
            results.push((name.to_string(), seed, summarize(&records, &config, seed)));
        }
        println!("done: {name}");
    }

    for seed in EVAL_START..EVAL_START + EVAL_PATHS {
        let mut config = base_config.clone();
        config.seed = seed;
        let cex_prices = generate_gbm(
            config.n_steps,
            INITIAL_PRICE,
            config.mu,
            config.sigma,
            dt,
            seed,
        );
        let records = run_simulation(&config, &cex_prices, &mut tabular_rl);
        results.push((
            "tabular_rl".to_string(),
            seed,
            summarize(&records, &config, seed),
        ));
    }
    println!("done: tabular_rl");

    // ── Compute paired delta vs oracle_gap ────────────────────────────────────
    let oracle_by_seed: HashMap<u64, usize> = results
        .iter()
        .enumerate()
        .filter(|(_, (n, _, _))| n == "oracle_gap")
        .map(|(i, (_, seed, _))| (*seed, i))
        .collect();

    // ── Write compare CSV ─────────────────────────────────────────────────────
    std::fs::create_dir_all("data/processed").unwrap();
    let compare_path = format!(
        "data/processed/campbell_rl_fee_compare_{}.csv",
        base_config.name
    );
    {
        let mut f = std::fs::File::create(&compare_path).unwrap();
        writeln!(
            f,
            "scenario,policy,seed,avg_fee_bps,hedged_pnl,lp_vs_hold,fee_revenue,lvr,\
             volume,arb_count,fundamental_count,final_external_price,final_pool_price,\
             normal_hedged_pnl,toxic_hedged_pnl,normal_steps,toxic_steps"
        )
        .unwrap();
        for (name, seed, m) in &results {
            writeln!(
                f,
                "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{:.6},{:.6},{:.4},{:.4},{},{}",
                base_config.name,
                name,
                seed,
                m.avg_fee_bps,
                m.hedged_pnl,
                m.lp_vs_hold,
                m.fee_revenue,
                m.lvr,
                m.volume,
                m.arb_count,
                m.fundamental_count,
                m.final_external_price,
                m.final_pool_price,
                m.normal_hedged_pnl,
                m.toxic_hedged_pnl,
                m.normal_steps,
                m.toxic_steps,
            )
            .unwrap();
        }
    }
    println!("\nSaved → {compare_path}");

    // ── Write paired delta CSV ────────────────────────────────────────────────
    let delta_path = format!(
        "data/processed/campbell_rl_paired_delta_{}.csv",
        base_config.name
    );
    {
        let mut f = std::fs::File::create(&delta_path).unwrap();
        writeln!(
            f,
            "scenario,policy,seed,\
             delta_hedged_pnl,delta_fee_revenue,delta_lvr,delta_volume,\
             delta_normal_hedged_pnl,delta_toxic_hedged_pnl"
        )
        .unwrap();
        for (name, seed, m) in &results {
            if name == "oracle_gap" {
                continue;
            }
            if let Some(&og_idx) = oracle_by_seed.get(seed) {
                let og = &results[og_idx].2;
                writeln!(
                    f,
                    "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    base_config.name,
                    name,
                    seed,
                    m.hedged_pnl - og.hedged_pnl,
                    m.fee_revenue - og.fee_revenue,
                    m.lvr - og.lvr,
                    m.volume - og.volume,
                    m.normal_hedged_pnl - og.normal_hedged_pnl,
                    m.toxic_hedged_pnl - og.toxic_hedged_pnl,
                )
                .unwrap();
            }
        }
    }
    println!("Saved → {delta_path}");

    // ── Summary table ─────────────────────────────────────────────────────────
    let has_regimes = base_config.flow_regime != FlowRegime::Normal;
    println!("\n{}", "─".repeat(60));
    println!(
        "Scenario: {}  ({} eval paths)",
        base_config.name, EVAL_PATHS
    );
    println!("{}", "─".repeat(60));
    println!(
        "{:<15} {:>10} {:>8} {:>8} {:>10} {:>10}",
        "policy", "mean_pnl", "p05_pnl", "fee_bps", "fee_rev", "lvr"
    );
    println!("{}", "─".repeat(60));

    for policy_name in &[
        "fixed_6bps",
        "fixed_10bps",
        "oracle_gap",
        "inventory_gap",
        "tabular_rl",
    ] {
        let rows: Vec<&PathMetrics> = results
            .iter()
            .filter(|(n, _, _)| n == policy_name)
            .map(|(_, _, m)| m)
            .collect();
        if rows.is_empty() {
            continue;
        }

        let mut pnls: Vec<f64> = rows.iter().map(|m| m.hedged_pnl).collect();
        let p05 = percentile(&mut pnls, 5.0);

        println!(
            "{:<15} {:>10.4} {:>8.4} {:>8.2} {:>10.4} {:>10.4}",
            policy_name,
            mean(&rows.iter().map(|m| m.hedged_pnl).collect::<Vec<_>>()),
            p05,
            mean(&rows.iter().map(|m| m.avg_fee_bps).collect::<Vec<_>>()),
            mean(&rows.iter().map(|m| m.fee_revenue).collect::<Vec<_>>()),
            mean(&rows.iter().map(|m| m.lvr).collect::<Vec<_>>()),
        );
    }

    if has_regimes {
        println!("\n{}", "─".repeat(60));
        println!("Regime breakdown  (normal vs toxic/arb-heavy)");
        println!("{}", "─".repeat(60));
        println!(
            "{:<15} {:>12} {:>12} {:>12} {:>12}",
            "policy", "norm_pnl", "toxic_pnl", "norm_frev", "toxic_frev"
        );
        println!("{}", "─".repeat(60));

        for policy_name in &["oracle_gap", "tabular_rl"] {
            let rows: Vec<&PathMetrics> = results
                .iter()
                .filter(|(n, _, _)| n == policy_name)
                .map(|(_, _, m)| m)
                .collect();
            if rows.is_empty() {
                continue;
            }
            println!(
                "{:<15} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                policy_name,
                mean(&rows.iter().map(|m| m.normal_hedged_pnl).collect::<Vec<_>>()),
                mean(&rows.iter().map(|m| m.toxic_hedged_pnl).collect::<Vec<_>>()),
                mean(
                    &rows
                        .iter()
                        .map(|m| m.normal_fee_revenue)
                        .collect::<Vec<_>>()
                ),
                mean(&rows.iter().map(|m| m.toxic_fee_revenue).collect::<Vec<_>>()),
            );
        }
    }

    // Paired delta vs oracle_gap
    println!("\n{}", "─".repeat(60));
    println!("Paired delta vs oracle_gap");
    println!("{}", "─".repeat(60));
    println!(
        "{:<15} {:>12} {:>12} {:>8}",
        "policy", "Δhedged_pnl", "Δfee_rev", "Δvolume"
    );
    println!("{}", "─".repeat(60));

    for policy_name in &["fixed_6bps", "fixed_10bps", "inventory_gap", "tabular_rl"] {
        let deltas: Vec<(f64, f64, f64)> = results
            .iter()
            .filter(|(n, _, _)| n == policy_name)
            .filter_map(|(_, seed, m)| {
                oracle_by_seed.get(seed).map(|&oi| {
                    let og = &results[oi].2;
                    (
                        m.hedged_pnl - og.hedged_pnl,
                        m.fee_revenue - og.fee_revenue,
                        m.volume - og.volume,
                    )
                })
            })
            .collect();
        if deltas.is_empty() {
            continue;
        }
        let pct_beat = deltas.iter().filter(|&&(d, _, _)| d > 0.0).count() as f64
            / deltas.len() as f64
            * 100.0;
        println!(
            "{:<15} {:>12.4} {:>12.4} {:>7.1} ({:.0}% beat)",
            policy_name,
            mean(&deltas.iter().map(|&(d, _, _)| d).collect::<Vec<_>>()),
            mean(&deltas.iter().map(|&(_, f, _)| f).collect::<Vec<_>>()),
            mean(&deltas.iter().map(|&(_, _, v)| v).collect::<Vec<_>>()),
            pct_beat,
        );
    }
    println!();
}
