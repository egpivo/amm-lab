# amm-lab

A Rust AMM simulation lab for studying execution mechanics, fee design, and LP economics.

## Scenarios

### Controlled pool scenarios

Run all four scenarios:

```bash
scripts/run_all_scenarios.sh
```

| Scenario | TOML | Description |
|---|---|---|
| Price impact ladder | `scenarios/price_impact_ladder.toml` | Execution drag across trade sizes on a 1T/1T pool |
| Same price, different depth | `scenarios/same_price_different_depth.toml` | Same spot price, three pool depths, same trade |
| Arbitrage repricing | `scenarios/arbitrage_repricing.toml` | Ternary-search arb closes price gap to external reference |
| LP vs hold | `scenarios/lp_vs_hold.toml` | LP withdrawal value vs passive hold after a 50% price move |

Artifacts written to `data/processed/`.

---

### Campbell et al. (2025) model

Implementation of the reduced-form CEX + DEX model from:

> Campbell, J. et al. (2025). *Optimal Fees for Liquidity Provision in AMMs.*

Three agents: arbitrageurs (closed-form trade sizing), fundamental buyers/sellers (routed optimally between AMM and CEX), and a passive LP collecting fees subject to LVR.

```bash
cargo run --bin campbell_sim scenarios/campbell_sim.toml
```

Key parameters in `scenarios/campbell_sim.toml`:

| Parameter | Default | Description |
|---|---|---|
| `amm_fee` | 0.003 | AMM fee η¹ (30 bps) |
| `cex_fee` | 0.001 | CEX fee η⁰ (10 bps) |
| `buy_demand` | 100.0 | Fundamental buy demand per step (Y units) |
| `sell_demand` | 100.0 | Fundamental sell demand per step (Y units) |
| `reserve_x` | 1_000_000.0 | Initial X reserve |
| `reserve_y` | 500.0 | Initial Y reserve (initial price = 2000) |
| `sigma` | 0.04 | CEX price annualized volatility |
| `mu` | 0.0 | CEX price drift |
| `n_steps` | 1440 | Simulation steps (1440 = 1 day at 1-min resolution) |
| `seed` | 42 | RNG seed for reproducibility |

Output artifacts:

| File | Description |
|---|---|
| `data/processed/campbell_sim.json` | Summary: fee revenue, tracking error, hedged PnL |
| `data/processed/campbell_sim_steps.csv` | Per-step: CEX price, AMM price, arb/buy/sell deltas, pool value, hedging portfolio |

**Key metrics:**

- **Hedged PnL** = Total fee revenue − Tracking error (LVR)
- **Tracking error** = Hedging portfolio − Pool value
- Positive hedged PnL means fees outpaced LVR on that path

---

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
