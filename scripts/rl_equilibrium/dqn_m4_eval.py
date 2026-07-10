"""M4: append DQN rows to m4_lp_adaptation.csv / m4_jit_mev.csv.

Frozen agent-first checkpoint (dqn_dynamic_duopoly.pt), ForcedTerminal,
reserved seed blocks LP 95_000-95_499 / JIT 96_000-96_499. The DQN was
trained under frozen liquidity and no JIT; these are transfer stress
evaluations by design.
"""

from common import OUT, append_csv_rows, read_csv
from dqn_core import greedy_episode, load_q_network
from gym_env import AmmExecutionEnv

FT = {"completion_rule": "forced_terminal"}


def main() -> None:
    grids = [
        (
            "m4_lp_adaptation.csv",
            "lp",
            "lp_regime",
            ["frozen", "weak", "aggressive"],
            95_000,
        ),
        ("m4_jit_mev.csv", "jit", "jit_regime", ["none", "weak", "aggressive"], 96_000),
    ]
    with AmmExecutionEnv(mode="dynamic_duopoly") as env:
        net = load_q_network(env, "dynamic_duopoly")
        for fname, extension, key, regimes, base in grids:
            path = OUT / fname
            if any(row["policy"] == "dqn" for row in read_csv(path)):
                raise SystemExit(
                    f"{fname} already contains DQN rows; "
                    "regenerate it with rl_equilibrium_sensitivity before appending"
                )
            rows = []
            for regime in regimes:
                agg = [0.0, 0.0]
                for seed in range(base, base + 500):
                    summary = greedy_episode(
                        env, net, seed, "dynamic_duopoly", **{key: regime}, **FT
                    )
                    rows.append(
                        [
                            extension,
                            regime,
                            "dqn",
                            seed,
                            f"{summary['shortfall_bps']:.4f}",
                            f"{summary['completion_rate']:.6f}",
                            f"{summary['fee_paid_bps']:.4f}",
                            f"{summary['gas_paid_bps']:.4f}",
                            f"{summary['slippage_ex_fee_bps']:.4f}",
                            f"{summary['drift_bps']:.4f}",
                            f"{summary['forced_terminal_cost_bps']:.4f}",
                            f"{summary['route_share_a']:.4f}",
                            f"{summary['route_share_b']:.4f}",
                            f"{summary['wait_share']:.4f}",
                            f"{summary['avg_depth_factor']:.4f}",
                            f"{summary['min_depth_factor']:.4f}",
                            summary["jit_event_count"],
                        ]
                    )
                    agg[0] += summary["shortfall_bps"]
                    agg[1] += summary["completion_rate"]
                print(
                    f"{extension}/{regime:<12} dqn IS {agg[0] / 500:8.2f} "
                    f"comp {agg[1] / 500:.4f}",
                    flush=True,
                )
            append_csv_rows(path, rows)


if __name__ == "__main__":
    main()
