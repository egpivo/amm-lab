# RL Execution / Routing (equilibrium-consistent simulator)

Closed-loop execution environment in `src/sim/`. Python DQN training talks to
`rl_equilibrium_bridge` over stdin/stdout JSON. Paper artifacts (CSVs,
checkpoints, figures, `m3r_run_manifest.json`) live in
`data/rl_equilibrium/` at the repo root — same layout as `data/causality/`
for the causal paper.

Paper notes (local): `.local/rl_equilibrium/plan.md`

## Quick start

```bash
# from repo root
cargo build --release --bin rl_equilibrium_bridge
pip install -r scripts/rl_equilibrium/requirements.txt

# smoke test
cd scripts/rl_equilibrium
python3 gym_env.py

# one deterministic episode
../../target/release/rl_equilibrium_sim --mode dynamic_duopoly --policy twap --seed 42
```

## Makefile pipeline

```bash
make -C scripts/rl_equilibrium help
make -C scripts/rl_equilibrium train-dqn    # ~15 min per checkpoint
make -C scripts/rl_equilibrium m3r-final
make -C scripts/rl_equilibrium verify       # checks data/rl_equilibrium/
```

## Binary map (v0 names → current)

| Paper / v0 name | Current binary |
|---|---|
| `export_rl_env` | `rl_equilibrium_bridge` |
| `run_execution_sim` | `rl_equilibrium_sim` |
| `run_m3_value_boundary` | `rl_equilibrium_ladder` |
| `run_m2_diagnostics` | `rl_equilibrium_artifact_battery` |
| `run_m3r_planner` | `rl_equilibrium_planner` |
| `run_baselines` | removed → `rl_equilibrium_train_tabular` |

All RL-equilibrium runners use the `rl_equilibrium_*` prefix. `campbell_*`
binaries belong to the Campbell / causal track.

## Python modules

| File | Role |
|---|---|
| `gym_env.py` | Gymnasium-style wrapper around the bridge |
| `dqn_core.py` | MLP Q-net, normalization, checkpoint load |
| `dqn_train.py` | DQN training + checkpoint selection |
| `dqn_diagnostics.py` | M3 artifact cells for DQN |
| `dqn_m3r_eval.py` | M3R evaluation matrices |
| `dqn_final_block.py` | Frozen 90k–90,999 seed block |
| `dqn_m4_eval.py` | M4 LP/JIT sensitivity append |
| `verify_paper_artifacts.py` | Assert headline numbers in `data/rl_equilibrium/` |
| `make_manifest.py` | `data/rl_equilibrium/m3r_run_manifest.json` |

Dependencies: `requirements.txt` (torch 2.5.1, numpy, matplotlib).

## Market modes

- `constant_duopoly` — two pools, fixed 30 bps (control)
- `dynamic_monopoly` — single pool, linear dynamic-fee rule
- `dynamic_duopoly` — two pools, each runs the rule against the other

## M1/M2 (tabular learner + diagnostics)

```bash
../../target/release/rl_equilibrium_train_tabular \
  --train-episodes 2000000 --n-val 200 --n-test 500

../../target/release/rl_equilibrium_artifact_battery --n-seeds 300 --n-fresh 500

python3 m1_m2_figures.py
```

## M3 (DQN + value boundary)

```bash
../../target/release/rl_equilibrium_train_tabular \
  --train-episodes 5000000 --spec fine --out-prefix m3_fine

../../target/release/rl_equilibrium_ladder --n-seeds 300

python3 dqn_train.py --episodes 12000
python3 dqn_diagnostics.py --n-seeds 300
python3 m3_figures.py
```

Headline (DynamicDuopoly, 500 test seeds): DQN **85.9 bps** vs tuned lookahead
**99.8** (paired −13.9 bps). Fine tabular ties lookahead.

## M3R (reviewer round)

```bash
make -C scripts/rl_equilibrium m3r-final
make -C scripts/rl_equilibrium figures
make -C scripts/rl_equilibrium manifest
make -C scripts/rl_equilibrium verify
```

Final block (forced terminal, seeds 90 000–90 999): agent-last DQN **100.3**
vs lookahead **113.6** bps.
