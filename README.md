# amm-lab

A Rust AMM simulation lab for studying execution mechanics, fee design, and LP economics under the Campbell et al. (2025) and Baggiani et al. (2025) frameworks.

## Scenarios

All scenario TOML files, parameter reference, and output format live in **[`scenarios/README.md`](scenarios/README.md)**.

| Scenario | Run |
|---|---|
| All four controlled-pool scenarios | `scripts/run_all_scenarios.sh` |
| Single controlled-pool scenario | `cargo run --release -- scenario run scenarios/<name>.toml` |

## Campbell et al. (2025) — Optimal Fee Model

Three binaries implement the [Campbell et al. (2025)](https://doi.org/10.2139/ssrn.4659452) reduced-form model of CEX + DEX with arbitrageurs and fundamental traders. Fees go to a separate LP account (pool reserves maintain constant product exactly). The model predicts an optimal AMM fee slightly below the CEX fee.

| Step | Binary | Output |
|---|---|---|
| 1 — Single path fee sweep | `cargo run --release --bin campbell_fee_sweep` | `data/processed/campbell_fee_sweep.csv` |
| 2 — Monte Carlo (500 paths) | `cargo run --release --bin campbell_monte_carlo` | `data/processed/campbell_monte_carlo.csv` |
| Single path detail | `cargo run --bin campbell_sim scenarios/campbell_sim.toml` | `data/processed/campbell_sim.json`, `data/processed/campbell_sim_steps.csv` |

Config: `scenarios/campbell_sim.toml` (shared across all three binaries).

**Key result:** with `cex_fee = 10 bps`, `avg_hedged_pnl` peaks at `fee_bps = 6`, consistent with the paper's prediction that the optimal AMM fee lies slightly below the CEX fee.

## Dynamic Fee Policies — Baggiani-style

Policy comparison across five fee strategies on 500 Monte Carlo paths.

| Binary | Output |
|---|---|
| `cargo run --release --bin campbell_dynamic_fee_sim` | `data/processed/campbell_sim_compare.csv`, `campbell_dynamic_fee_steps.csv` |
| `cargo run --bin campbell_policy_audit` | `data/processed/campbell_dynamic_fee_policy_audit.csv` |

Policies: `fixed_6bps`, `fixed_10bps`, `oracle_gap`, `inventory_gap`.

**Key result:** `OracleGapFeePolicy` (base 6 bps + 0.1 × |gap_bps| / 10000) outperforms all fixed policies. `inventory_skew` and `oracle_gap` are nearly collinear in CPMM geometry (Pearson r ≈ 1.0), so `InventoryGapFeePolicy` adds little beyond fixed_6bps.

## Naive RL Fee Policy (v0 / v0.1)

Tabular Monte Carlo Q-learning over a 72-state space (gap × vol × flow).
Trained on 5,000 GBM paths; evaluated on 500 held-out paths.

| Binary | Output |
|---|---|
| `cargo run --release --bin campbell_rl_fee_train` | `data/processed/campbell_rl_fee_table.csv`, `campbell_rl_training_summary.csv` |
| `cargo run --bin campbell_rl_fee_compare` | `data/processed/campbell_rl_fee_compare.csv`, `campbell_rl_paired_delta_vs_oracle_gap.csv` |
| `cargo run --bin campbell_rl_q_diagnostics` | Q-table monotonicity analysis (stdout) |

| policy | mean hedged_pnl | mean fee (bps) |
|---|---|---|
| oracle_gap | **0.4239** | 6.88 |
| fixed_6bps | 0.4088 | 6.00 |
| naive_rl v0.1 | 0.3118 | 5.85 |
| naive_rl v0 | 0.2762 | 16.19 |

**Key findings:**
- v0 (unconstrained actions to 100 bps): overcharges, kills ~590 units of fundamental volume per path, collects less fee revenue than fixed_6bps.
- v0.1 (actions capped at 15 bps): fixes fee level but not policy shape. 40.6% of gap transitions are non-monotone; weighted violation rate 29.9%.
- Oracle-gap heuristic remains the strongest baseline. Naive tabular RL with terminal reward does not recover the monotone stale-price defense structure from scratch.
- Next direction: monotone projection, oracle-gap warm start, or paired reward signal.

## Module map

```
src/
├── pool.rs          # reserves, fee bps, invariant checks (u128)
├── swap.rs          # exact-input quote and execution
├── liquidity.rs     # LP mint/burn accounting
├── arbitrage.rs     # ternary-search profit-maximizing arb
├── scenario.rs      # TOML scenario runner
├── lp_accounting.rs # LP-vs-hold report
└── campbell/
    ├── pool.rs      # f64 CPMM with separate fee accounting (k maintained exactly)
    ├── trader.rs    # arb_delta, fundamental_buy_delta, fundamental_sell_delta
    ├── gbm.rs       # GBM price path generator
    ├── simulation.rs # per-step loop, rolling vol/flow window, StepRecord
    └── fee_policy.rs # FeePolicy trait, fixed/oracle/inventory/tabular-RL policies
```

## Build

```bash
cargo build
cargo test --all
```

Requires Rust 2024 edition.
