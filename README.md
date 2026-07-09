# amm-lab

A Rust AMM simulation lab and empirical toolkit for fee design and LP economics. This repo
holds the code, data, and reproducible figures behind the accompanying blog posts and
research papers, building on the fee-design frameworks of
[Campbell et al. (2025)](https://doi.org/10.2139/ssrn.4659452) and Baggiani et al. (2025).

## Simulation

A structural CPMM environment (CEX + DEX, arbitrageurs and fundamental traders) for
do-intervention fee-policy contrasts. Scenario files and parameters live in
**[`scenarios/README.md`](scenarios/README.md)**; outputs land in `data/processed/`.

| Binaries | Purpose |
|---|---|
| `campbell_fee_sweep`, `campbell_monte_carlo` | optimal static fee (Campbell et al. 2025) |
| `campbell_dynamic_fee_compare`, `campbell_policy_audit` | dynamic-fee policy comparison (Baggiani-style) |

Run with `cargo run --release --bin <name>`. A fee policy is an intervention `do(Π=π)`
and effects are paired common-random-number contrasts. The identification boundary:
fixed-fee v3 history cannot identify dynamic-fee counterfactuals, so those are identified
only inside the simulator, which empirical replay calibrates. Full framing and results are
in the blog posts and papers.

## Empirical paper — protocol-fee switch (causal inference)

Matched-overlap event study of the Uniswap protocol-fee switch (2025-12-28): no large
short-run LP-supply/depth response; flow/revenue fail the gate; dynamic-fee protection not
identified.

- **Scripts:** `scripts/causality/` — `analysis_es.R` (core), `robustness_battery.R`, `honest_did.R`, `fpca_diagnostic.R`, `fect_vol1.R`, `figures/fig01–05`.
- **Data:** `data/causality/` — panel, matched pairs, activation events, estimation outputs.

```bibtex
@unpublished{wang2026protocolfee,
  author = {Wen-Ting Wang},
  title  = {Causal Effects of Protocol-Fee Changes on Liquidity Provision in Automated Market Makers},
  year   = {2026}, note = {Working paper}
}
```

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
