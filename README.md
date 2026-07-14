# amm-lab

Rust lab for AMM execution mechanics, on-chain causal identification, and
closed-loop RL simulation. The repo holds **three tracks**:

| # | Track | Status |
|---|---|---|
| 1 | **AMM scenarios** — controlled-pool mechanics (practice) | stable |
| 2 | **Paper — causality** — Uniswap protocol-fee switch, channel framework | [arXiv:2607.08525](https://arxiv.org/abs/2607.08525) |
| 3 | **Paper — RL equilibrium** — execution routing in a dynamic-fee duopoly | [arXiv:2607.10960](https://arxiv.org/abs/2607.10960) |

## Build

```bash
make build
make test
make help
```

Requires Rust 2024 edition. Operational detail for each paper track lives in
its scripts README (below).

---

## 1. AMM scenarios (practice)

Exact `u128` CPMM: reserves, swaps, liquidity mint/burn, arbitrage, LP-vs-hold.
No market response — isolated mechanics.

**Docs:** [`scenarios/README.md`](scenarios/README.md)

```bash
make scenarios
cargo run --release -- scenario run scenarios/<name>.toml
```

Core code: `src/pool.rs`, `swap.rs`, `liquidity.rs`, `arbitrage.rs`, `scenario.rs`.

---

## 2. Paper — causality

Event-study and channel-audit tooling for the protocol-fee-switch paper.
Historical identification of LP-supply response (K_L); Campbell et al. (2025)
reduced-form model appears as a **compressed simulation diagnostic**, not the
empirical estimand.

- **Paper:** [Causal Effects of Protocol-Fee Changes on Liquidity Provision in Automated Market Makers](https://arxiv.org/abs/2607.08525) ([pdf](https://arxiv.org/pdf/2607.08525))
- **Ops:** [`scripts/causality/README.md`](scripts/causality/README.md)

```bibtex
@misc{wang2026protocolfee,
  title         = {Causal Effects of Protocol-Fee Changes on Liquidity Provision in Automated Market Makers},
  author        = {Wang, Wen-Ting},
  year          = {2026},
  eprint        = {2607.08525},
  archivePrefix = {arXiv},
  primaryClass  = {stat.AP},
  url           = {https://arxiv.org/abs/2607.08525}
}
```

| Layer | Location |
|---|---|
| Event study / panel | `event_study`, `panel_report`, `panel_compare` |
| Estimation scripts | `scripts/causality/` |
| Channel audit | `src/audit/`, `src/causal/` |
| On-chain data | `src/data/`, `data/causality/` |
| Model-conditioned sim | `src/campbell/`, `campbell_*` binaries, `scenarios/campbell_*.toml` |

---

## 3. Paper — RL equilibrium

Closed-loop dynamic-fee duopoly: an execution agent's trades move inventory,
quotes, fees, and arbitrage. PyTorch DQN trains through a Rust JSON bridge.

- **Paper:** [Reinforcement Learning for Execution under Dynamic Fees in a Closed-Loop DEX Simulator](https://arxiv.org/abs/2607.10960) ([pdf](https://arxiv.org/pdf/2607.10960))
- **Ops:** [`scripts/rl_equilibrium/README.md`](scripts/rl_equilibrium/README.md)

```bibtex
@misc{wang2026rldex,
  title         = {Reinforcement Learning for Execution under Dynamic Fees in a Closed-Loop {DEX} Simulator},
  author        = {Wang, Wen-Ting},
  year          = {2026},
  eprint        = {2607.10960},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.10960}
}
```

| Layer | Location |
|---|---|
| Simulator | `src/sim/` |
| Rust runners | `rl_equilibrium_*` binaries |
| DQN pipeline | `scripts/rl_equilibrium/` |
| Paper artifacts | `data/rl_equilibrium/` (CSVs, checkpoints, figures, manifest) |

---

## Module map

```
src/
├── pool.rs, swap.rs, scenario.rs …     # (1) AMM scenarios
├── causal/, data/, audit/              # (2) causality paper
├── campbell/                           # (2) model-conditioned diagnostic
├── sim/                                # (3) RL-equilibrium env
scripts/
├── causality/                          # (2) ops → README.md
├── rl_equilibrium/                     # (3) ops → README.md
data/
├── causality/                          # (2) on-chain panels & analysis
├── rl_equilibrium/                     # (3) paper artifacts
scenarios/                              # (1) + campbell TOMLs for (2)
```
