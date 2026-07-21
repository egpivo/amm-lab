# amm-lab

Rust lab for AMM execution mechanics, on-chain causal identification,
closed-loop RL simulation, and LVR measurement under programmable hooks.
The repo holds **four tracks**:

| # | Track | Status |
|---|---|---|
| 1 | **AMM scenarios** — controlled-pool mechanics (practice) | stable |
| 2 | **Paper — causality** — Causal Effects of Protocol-Fee Changes on Liquidity Provision in Automated Market Makers | [arXiv:2607.08525](https://arxiv.org/abs/2607.08525) |
| 3 | **Paper — RL equilibrium** — Reinforcement Learning for Execution under Dynamic Fees in a Closed-Loop DEX Simulator | [arXiv:2607.10960](https://arxiv.org/abs/2607.10960) |
| 4 | **Paper — LVR** — Policy-Selected Transaction Tapes in Automated Market Makers: Ranking Certification under Hidden Opportunities | arXiv TBD (title may change) |

## Build

```bash
make build
make test
make help
```

Requires Rust 2024 edition.

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

**Causal Effects of Protocol-Fee Changes on Liquidity Provision in Automated
Market Makers**
([arXiv:2607.08525](https://arxiv.org/abs/2607.08525)).

Event-study and channel-audit tooling for the protocol-fee-switch paper.
Historical identification of LP-supply response (K_L); Campbell et al. (2025)
reduced-form model appears as a **compressed simulation diagnostic**, not the
empirical estimand.

| Layer | Location |
|---|---|
| Event study / panel | `event_study`, `panel_report`, `panel_compare` |
| Estimation scripts | `scripts/causality/` |
| Channel audit | `src/audit/`, `src/causal/` |
| On-chain data | `src/data/`, `data/causality/` |
| Model-conditioned sim | `src/campbell/`, `campbell_*` binaries, `scenarios/campbell_*.toml` |

```bash
# example: event-study coefficient path
cargo run --release --bin event_study -- --estimate --out data/causality/analysis_r_cal0.25

# Campbell diagnostic (optimal fee under reduced-form CEX+DEX)
cargo run --release --bin campbell_fee_sweep
cargo run --release --bin campbell_monte_carlo
```

Exploratory fee-policy sims (oracle-gap heuristics, tabular RL) live under
`campbell_rl_*` and support the paper's identification-boundary discussion;
they are not a separate paper track.

---

## 3. Paper — RL equilibrium

**Reinforcement Learning for Execution under Dynamic Fees in a Closed-Loop
DEX Simulator**
([arXiv:2607.10960](https://arxiv.org/abs/2607.10960)).

Closed-loop dynamic-fee duopoly: an execution agent's trades move inventory,
quotes, fees, and arbitrage. PyTorch DQN trains through a Rust JSON bridge.

| Layer | Location |
|---|---|
| Simulator | `src/sim/` |
| Rust runners | `rl_equilibrium_*` binaries |
| DQN pipeline | `scripts/rl_equilibrium/` |
| Paper artifacts | `data/rl_equilibrium/` (CSVs, checkpoints, figures, manifest) |

```bash
pip install -r scripts/rl_equilibrium/requirements.txt
make -C scripts/rl_equilibrium help
make -C scripts/rl_equilibrium verify    # checks data/rl_equilibrium/
make -C scripts/rl_equilibrium train-dqn  # regenerates into data/rl_equilibrium/
```

**Docs:** [`scripts/rl_equilibrium/README.md`](scripts/rl_equilibrium/README.md)

---

## 4. Paper — LVR

**Policy-Selected Transaction Tapes in Automated Market Makers: Ranking
Certification under Hidden Opportunities**
(arXiv TBD; title may change).

Policy-selected LVR under programmable hooks: latent opportunities as a
marked point process, hooks as predictable output kernels, and ranking /
certification limits when the mechanism also selects the observed tape.

| Layer | Location |
|---|---|
| LVR runners | `lvr_*` binaries (`src/bin/lvr_*.rs`) |
| Analysis scripts | `scripts/lvr/` |
| Shared sim engine | `src/campbell/` (arrival / fee-policy simulation) |
| PSTT / tape tooling | `src/pstt/`, `pstt_*` binaries |
| Working manuscript | `.local/lvr/` (gitignored) |

```bash
# example: paired decomposition / grid diagnostics
cargo run --release --bin lvr_paired_decomposition
cargo run --release --bin lvr_grid_decomposition
```

---

## Module map

```
src/
├── pool.rs, swap.rs, scenario.rs …     # (1) AMM scenarios
├── causal/, data/, audit/              # (2) causality paper
├── campbell/                           # (2) diagnostic + (4) LVR sim engine
├── sim/                                # (3) RL-equilibrium env
├── pstt/                               # (4) LVR / policy-selected tapes
scripts/
├── causality/                          # (2)
├── rl_equilibrium/                     # (3)
├── lvr/                                # (4)
data/
├── causality/                          # (2) on-chain panels & analysis
├── rl_equilibrium/                     # (3) paper artifacts
scenarios/                              # (1) + campbell TOMLs for (2)
```
