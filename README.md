# amm-lab

A Rust AMM simulation lab for studying execution mechanics, fee design, and LP economics.

## Scenarios

All scenario TOML files, parameter reference, and output format live in **[`scenarios/README.md`](scenarios/README.md)**.

| Scenario | Run |
|---|---|
| All four controlled-pool scenarios | `scripts/run_all_scenarios.sh` |
| Single controlled-pool scenario | `cargo run --release -- scenario run scenarios/<name>.toml` |
| Campbell et al. (2025) model | `cargo run --bin campbell_sim scenarios/campbell_sim.toml` |

Artifacts: `data/processed/`.

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
    ├── pool.rs      # f64 CPMM with separate fee accounting
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
