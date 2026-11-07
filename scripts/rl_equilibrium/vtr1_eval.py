"""VTR-1 DQN evaluation of a frozen checkpoint on one seed block under one scenario.

Pilot and final blocks need an authorization file (seed_blocks). The output CSV must not
exist and is never one of the archived artifacts. Training and kappa tuning use the same
scenario file (scenario_config).
"""

import argparse
from pathlib import Path

import scenario_config
import seed_blocks


FORCED_TERMINAL = "forced_terminal"


def plan(args: argparse.Namespace) -> tuple[seed_blocks.Block, dict, tuple[str, str]]:
    """Validate everything that can be validated without starting the simulator."""
    block = seed_blocks.require(args.role, args.seed_base, args.n, args.authorization)
    mode, order = scenario_config.check_cell(args.cell)
    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"checkpoint {args.checkpoint} not found")
    seed_blocks.check_new_outputs([args.out])
    # forced-terminal completion is the VTR-1 evaluation rule; there is no way to change it
    overrides = {
        "agent_order": order,
        **scenario_config.overrides(args.scenario),
        "completion_rule": FORCED_TERMINAL,
    }
    return block, overrides, (mode, order)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cell", required=True, help="mode/order, e.g. dynamic_duopoly/before")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--role", choices=["eval", "pilot", "final"], required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--authorization", default=None)
    ap.add_argument("--replicate-id", default=None)
    ap.add_argument("--out", required=True)
    return ap


def main() -> None:
    args = parser().parse_args()
    block, overrides, (mode, _) = plan(args)

    import torch  # deferred: planning and guards need neither torch nor the simulator

    from common import write_csv
    from dqn_core import QNet, greedy_episode
    from gym_env import AmmExecutionEnv

    with AmmExecutionEnv(mode=mode) as env:
        net = QNet(env.obs_dim, env.n_actions)
        net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        net.eval()
        rows = []
        for seed in range(args.seed_base, args.seed_base + args.n):
            summary = greedy_episode(env, net, seed, mode, **overrides)
            rows.append(
                {
                    "cell": args.cell,
                    "scenario": scenario_config.load(args.scenario).get("name", "baseline"),
                    "replicate_id": args.replicate_id,
                    "block": block.name,
                    "seed": seed,
                    **summary,
                }
            )
    write_csv(Path(args.out), list(rows[0]), rows)


if __name__ == "__main__":
    main()
