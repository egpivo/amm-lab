# amm-lab

A Rust AMM simulation lab for studying execution mechanics, fee design, and LP economics.

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

### Output columns — Monte Carlo CSV

| Column | Description |
|---|---|
| `fee_bps` | AMM fee in basis points (1–100) |
| `amm_fee` | AMM fee as a fraction |
| `avg_hedged_pnl` | Mean hedged PnL = fee revenue − LVR, across 500 paths |
| `std_hedged_pnl` | Std dev of hedged PnL |
| `avg_lp_vs_hold` | Mean LP value (pool + fees) minus passive hold value |
| `std_lp_vs_hold` | Std dev of LP-vs-hold |

**Key result:** with `cex_fee = 10 bps`, `avg_hedged_pnl` peaks at `fee_bps = 6`, consistent with the paper's prediction that the optimal AMM fee lies slightly below the CEX fee.

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
    └── simulation.rs # per-step loop, StepRecord, SimSummary
```

## Build

```bash
cargo build
cargo test --all
```

Requires Rust 2024 edition.
