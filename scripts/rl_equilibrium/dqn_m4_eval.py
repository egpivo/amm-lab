"""M4: append DQN rows to m4_lp_adaptation.csv / m4_jit_mev.csv.

Frozen agent-first checkpoint (dqn_dynamic_duopoly.pt), ForcedTerminal,
reserved seed blocks LP 95_000-95_499 / JIT 96_000-96_499. The DQN was
trained under frozen liquidity and no JIT; these are transfer stress
evaluations by design.
"""

from __future__ import annotations

from pathlib import Path

import torch

from dqn_train import QNet, normalize, reset_env
from gym_env import AmmExecutionEnv

OUT = Path(__file__).resolve().parents[2] / "experiments/rl_execution/out"
FT = {"completion_rule": "forced_terminal"}


def main() -> None:
    env = AmmExecutionEnv(mode="dynamic_duopoly")
    net = QNet(env.obs_dim, env.n_actions)
    net.load_state_dict(torch.load(OUT / "dqn_dynamic_duopoly.pt", weights_only=True))
    net.eval()

    grids = [
        ("m4_lp_adaptation.csv", "lp", "lp_regime",
         ["frozen", "weak", "aggressive"], 95_000),
        ("m4_jit_mev.csv", "jit", "jit_regime",
         ["none", "weak", "aggressive"], 96_000),
    ]
    for fname, ext, key, regimes, base in grids:
        # idempotency guard: refuse to append twice (rerun rl_equilibrium_sensitivity first)
        with open(OUT / fname) as f:
            if any(",dqn," in line for line in f):
                raise SystemExit(f"{fname} already contains DQN rows; "
                                 "regenerate it with rl_equilibrium_sensitivity before appending")
        with open(OUT / fname, "a") as f:
            for regime in regimes:
                agg = [0.0, 0.0]
                for seed in range(base, base + 500):
                    obs = reset_env(env, seed, "dynamic_duopoly",
                                    **{key: regime}, **FT)
                    done, info = False, {}
                    while not done:
                        with torch.no_grad():
                            a = int(net(normalize(obs)).argmax())
                        obs, _, done, _, info = env.step(a)
                    s = info["summary"]
                    f.write(
                        f"{ext},{regime},dqn,{seed},{s['shortfall_bps']:.4f},"
                        f"{s['completion_rate']:.6f},{s['fee_paid_bps']:.4f},"
                        f"{s['gas_paid_bps']:.4f},{s['slippage_ex_fee_bps']:.4f},"
                        f"{s['drift_bps']:.4f},{s['forced_terminal_cost_bps']:.4f},"
                        f"{s['route_share_a']:.4f},{s['route_share_b']:.4f},"
                        f"{s['wait_share']:.4f},{s['avg_depth_factor']:.4f},"
                        f"{s['min_depth_factor']:.4f},{s['jit_event_count']}\n")
                    agg[0] += s["shortfall_bps"]
                    agg[1] += s["completion_rate"]
                print(f"{ext}/{regime:<12} dqn IS {agg[0]/500:8.2f} "
                      f"comp {agg[1]/500:.4f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
