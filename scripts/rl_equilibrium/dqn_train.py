"""Train and evaluate the validation-grid DQN through the Rust bridge."""

import argparse
import random
import time
from pathlib import Path
from collections import deque
from typing import TypeAlias

import torch
import torch.nn as nn

import sys

import seed_blocks
from common import OUT, write_csv
from dqn_core import QNet, evaluate, greedy_action, greedy_episode, normalize, reset_env
from gym_env import AmmExecutionEnv
from training_validity import TrainingInvalid, ValidityMonitor

ReplayItem: TypeAlias = tuple[torch.Tensor, int, float, torch.Tensor, bool]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--mode", default="dynamic_duopoly")
    parser.add_argument(
        "--agent-order", default="before", choices=["before", "random", "after"]
    )
    parser.add_argument("--train-penalty", type=float, default=0.02)
    parser.add_argument("--tag", default="dynamic_duopoly")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Framework seed for torch/random (independent training replicate).",
    )
    parser.add_argument("--train-seed-base", type=int, default=1_000_000)
    parser.add_argument("--val-seed-base", type=int, default=20_000)
    parser.add_argument("--test-seed-base", type=int, default=30_000)
    parser.add_argument("--fresh-seed-base", type=int, default=40_000)
    parser.add_argument(
        "--check-finite",
        action="store_true",
        help="VTR-1 technical training-validity checks; writes a per-run validity record.",
    )
    parser.add_argument("--validity-out", default=None)
    args = parser.parse_args()
    if min(args.episodes, args.eval_every, args.n_val, args.n_test) <= 0:
        parser.error("episode and seed counts must be positive")

    seed_blocks.require("train", args.train_seed_base, args.episodes)
    seed_blocks.require("val", args.val_seed_base, args.n_val)
    seed_blocks.require("eval", args.test_seed_base, args.n_test)
    seed_blocks.require("eval", args.fresh_seed_base, args.n_test)

    monitor = ValidityMonitor(args.check_finite)
    validity_path = (
        OUT / f"validity_{args.tag}.json"
        if args.validity_out is None
        else Path(args.validity_out)
    )
    try:
        run(args, monitor)
    except TrainingInvalid as error:
        monitor.write(validity_path, args.tag)
        print(f"TRAINING_INVALID: {error}", file=sys.stderr, flush=True)
        sys.exit(3)
    if args.check_finite:
        monitor.write(validity_path, args.tag)


def run(args: argparse.Namespace, monitor: ValidityMonitor) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    train_ov = {
        "agent_order": args.agent_order,
        "unfinished_penalty": args.train_penalty,
    }
    eval_ov = {"agent_order": args.agent_order}
    with AmmExecutionEnv(mode=args.mode) as env:
        net = QNet(env.obs_dim, env.n_actions)
        target = QNet(env.obs_dim, env.n_actions)
        target.load_state_dict(net.state_dict())
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.SmoothL1Loss()

        replay: deque[ReplayItem] = deque(maxlen=200_000)
        batch_size, warmup, sync_every = 64, 5_000, 2_000
        step_count = 0
        curve: list[dict[str, float | int]] = []
        best_value = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        val_seeds = range(args.val_seed_base, args.val_seed_base + args.n_val)
        started_at = time.monotonic()

        for episode in range(args.episodes):
            epsilon = max(0.05, 1.0 - 0.95 * episode / (0.6 * args.episodes))
            observation = reset_env(env, args.train_seed_base + episode, args.mode, **train_ov)
            state = normalize(observation)
            done = False
            while not done:
                action = (
                    random.randrange(env.n_actions)
                    if random.random() < epsilon
                    else greedy_action(net, observation)
                )
                observation, reward, done, _, _ = env.step(action)
                next_state = normalize(observation)
                replay.append((state, action, reward * 100.0, next_state, done))
                state = next_state
                step_count += 1

                if len(replay) >= warmup:
                    sample = random.sample(replay, batch_size)
                    states = torch.stack([item[0] for item in sample])
                    actions = torch.tensor([item[1] for item in sample])
                    rewards = torch.tensor([item[2] for item in sample])
                    next_states = torch.stack([item[3] for item in sample])
                    dones = torch.tensor([float(item[4]) for item in sample])
                    monitor.check_inputs(step_count, states, next_states, rewards)
                    values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.inference_mode():
                        targets = (
                            rewards + (1.0 - dones) * target(next_states).max(1).values
                        )
                    loss = loss_fn(values, targets.clone())
                    monitor.check_forward(step_count, values, targets, loss)
                    optimizer.zero_grad()
                    loss.backward()
                    monitor.check_grads(step_count, net)
                    optimizer.step()
                    monitor.check_after_step(step_count, net, optimizer)
                if step_count % sync_every == 0:
                    target.load_state_dict(net.state_dict())

            if (episode + 1) % args.eval_every == 0:
                value = evaluate(env, net, val_seeds, args.mode, **eval_ov)
                curve.append(
                    {"episode": episode + 1, "val_shortfall_bps": value, "eps": epsilon}
                )
                marker = ""
                if value < best_value:
                    best_value = value
                    best_state = {
                        key: tensor.detach().cpu().clone()
                        for key, tensor in net.state_dict().items()
                    }
                    marker = "  <- best"
                elapsed = time.monotonic() - started_at
                print(
                    f"ep {episode + 1:>6}  eps {epsilon:.2f}  "
                    f"val IS {value:8.2f} bps  ({elapsed:.0f}s){marker}",
                    flush=True,
                )

        if best_state is None:
            best_value = evaluate(env, net, val_seeds, args.mode, **eval_ov)
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in net.state_dict().items()
            }
            curve.append(
                {
                    "episode": args.episodes,
                    "val_shortfall_bps": best_value,
                    "eps": epsilon,
                }
            )

        net.load_state_dict(best_state)
        torch.save(net.state_dict(), OUT / f"dqn_{args.tag}.pt")
        write_csv(
            OUT / f"m3_dqn_curve_{args.tag}.csv",
            ["episode", "val_shortfall_bps", "eps"],
            curve,
        )

        results = []
        for label, base in (
            ("test", args.test_seed_base),
            ("fresh", args.fresh_seed_base),
        ):
            for seed in range(base, base + args.n_test):
                summary = greedy_episode(env, net, seed, args.mode, **eval_ov)
                results.append(
                    {"policy": f"dqn_{args.tag}", "seed_set": label, **summary}
                )
        write_csv(OUT / f"m3_dqn_results_{args.tag}.csv", list(results[0]), results)
    print("selected checkpoint val IS:", round(best_value, 2))


if __name__ == "__main__":
    main()
