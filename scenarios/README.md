# Scenarios

TOML files in this directory define simulation setups. Artifacts are written to `data/processed/`.

## Quick start

Run all four controlled-pool scenarios:

```bash
scripts/run_all_scenarios.sh
```

Run a single controlled-pool scenario:

```bash
cargo run --release -- scenario run scenarios/<name>.toml
```

Run the Campbell et al. (2025) model:

```bash
cargo run --bin campbell_sim scenarios/campbell_sim.toml
```

---

## Controlled pool scenarios

Each file is a declarative script: a sequence of `[[transactions]]` applied to a u128 CPMM pool.

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Scenario identifier; used as the output filename stem |
| `description` | string | Human-readable summary stored in the JSON report |
| `transactions` | array | Ordered list of transaction steps (see below) |

### Transaction types

All amounts are unsigned integers (`u64` in TOML, stored as `u128` internally).

| `type` | Fields | Description |
|---|---|---|
| `CreatePool` | `reserve_x`, `reserve_y`, `fee_bps` | Initialize a new pool. Resets any tracked LP position. Spot price = `reserve_y / reserve_x`. |
| `AddLiquidity` | `actor`, `amount_x`, `amount_y` | Mint LP shares proportional to pool reserves. Records the first LP position for later reporting. |
| `RemoveLiquidity` | `actor`, `lp_shares` | Burn LP shares and return pro-rata reserves. |
| `SwapExactIn` | `actor`, `direction`, `amount_in`, `min_amount_out` | Exact-input swap. `direction`: `XtoY` or `YtoX`. Reverts if output &lt; `min_amount_out`. |
| `ExternalPriceMove` | `new_price` | Set the external reference price (Y per X) used for arbitrage and LP valuation. |
| `ArbitrageUntilNoProfit` | `max_steps` | Run ternary-search arbitrage against `ExternalPriceMove` price until profit ≤ 0 or `max_steps` reached. |
| `ReportLpPerformance` | `actor` | Compare LP withdrawal value vs passive hold at the current external price. Requires a prior `AddLiquidity`. |

**Actors:** `Lp1`, `Trader1`, `Arbitrageur1`, `Pool` (labels only; no wallet logic).

### Output artifacts

For each controlled-pool run, three files are written to `data/processed/`:

| File | Contents |
|---|---|
| `{name}.json` | Full report: swap events, arbitrage summary, LP performance, execution log |
| `{name}_swaps.csv` | Per-swap: amounts, fees, exec price, price impact, reserves |
| `{name}_arbitrage.csv` | Per-arb-step: direction, profit estimate, price gap before/after |

---

### `price_impact_ladder.toml`

Execution drag across trade sizes on a 1T/1T pool at 30 bps fee.

Six independent sub-runs (each starts with `CreatePool`), trade sizes from 0.01% to 25% of pool depth:

| Sub-run | `amount_in` | % of pool |
|---|---|---|
| 1 | 100_000_000 | 0.01% |
| 2 | 1_000_000_000 | 0.1% |
| 3 | 10_000_000_000 | 1% |
| 4 | 50_000_000_000 | 5% |
| 5 | 100_000_000_000 | 10% |
| 6 | 250_000_000_000 | 25% |

Key parameters to tweak: `fee_bps`, `reserve_x` / `reserve_y` (pool depth), `amount_in` ladder.

---

### `same_price_different_depth.toml`

Same spot price (1.0), three pool depths, identical 10B X→Y trade.

| Pool | Reserves (X, Y) | Trade as % of pool |
|---|---|---|
| A (shallow) | 100B / 100B | 10% |
| B (medium) | 1T / 1T | 1% |
| C (deep) | 10T / 10T | 0.1% |

Key parameters: `reserve_x` / `reserve_y` per sub-run, shared `amount_in = 10_000_000_000`.

---

### `arbitrage_repricing.toml`

1T/1T pool at price 1.0. External price jumps to 1.5; arbitrage reprices the pool step by step.

| Step | Transaction | Key values |
|---|---|---|
| 1 | `CreatePool` | 1T / 1T, `fee_bps = 30` |
| 2 | `ExternalPriceMove` | `new_price = 1.5` |
| 3 | `ArbitrageUntilNoProfit` | `max_steps = 50` |

Key parameters: `new_price` (gap size), `fee_bps` (arb profitability), `max_steps`.

---

### `lp_vs_hold.toml`

LP deposits into a 1T/1T pool, traders generate fee income, external price moves +50%, arbitrage reprices, then LP vs hold is reported.

| Phase | Transactions | Notes |
|---|---|---|
| Setup | `CreatePool` + `AddLiquidity` | LP1 adds 500B X + 500B Y (50% of pool) |
| Volume | 3× `SwapExactIn` | 10B Y→X, 10B X→Y, 20B X→Y |
| Repricing | `ExternalPriceMove` + `ArbitrageUntilNoProfit` | `new_price = 1.5`, `max_steps = 50` |
| Report | `ReportLpPerformance` | actor `Lp1` |

**LP report fields** (in JSON):

- `fee_income_value_in_y` — pro-rata share of accumulated fees valued at external price (proxy)
- `hold_value_in_y` — value of initial deposit at external price
- `lp_value_in_y` — withdrawal value at current pool state
- `impermanent_loss_pct` — IL relative to hold
- `net_profit_loss_in_y` — LP value − hold value (fees included in pool reserves)

---

## Campbell et al. (2025) model

Implementation of the reduced-form CEX + DEX model from:

> Campbell, J. et al. (2025). *Optimal Fees for Liquidity Provision in AMMs.*

Three agents per step: arbitrageurs (closed-form sizing), fundamental buyers/sellers (routed between AMM and CEX), and a passive LP collecting fees subject to LVR.

Config file: [`campbell_sim.toml`](campbell_sim.toml)

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `name` | `campbell_sim` | Scenario identifier |
| `description` | — | Human-readable summary |
| `amm_fee` | 0.003 | AMM fee η¹ (30 bps) |
| `cex_fee` | 0.001 | CEX fee η⁰ (10 bps) |
| `buy_demand` | 100.0 | Fundamental buy demand per step (Y units) |
| `sell_demand` | 100.0 | Fundamental sell demand per step (Y units) |
| `reserve_x` | 1_000_000.0 | Initial X reserve |
| `reserve_y` | 500.0 | Initial Y reserve (initial AMM price = 2000) |
| `sigma` | 0.04 | CEX price annualized volatility |
| `mu` | 0.0 | CEX price drift |
| `n_steps` | 1440 | Simulation steps (1440 = 1 day at 1-min resolution) |
| `seed` | 42 | RNG seed for reproducibility |

CEX initial price is fixed at 2000.0 in the binary (matching `reserve_y / reserve_x` at defaults). Step interval = `1 / n_steps` (annualized GBM).

### Output artifacts

| File | Description |
|---|---|
| `data/processed/campbell_sim.json` | Summary: fee revenue, tracking error, hedged PnL, final prices |
| `data/processed/campbell_sim_steps.csv` | Per-step: CEX price, AMM price, arb/buy/sell deltas, pool value, hedging portfolio |

**Key metrics** (printed to stderr and stored in JSON):

- **Hedged PnL** = total fee revenue − tracking error (LVR)
- **Tracking error** = hedging portfolio − pool value
- Positive hedged PnL means fees outpaced LVR on that path
