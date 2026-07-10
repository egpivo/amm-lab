"""M3A: small DQN over the Rust bridge.

Design (recorded for m3_learner.md):
- observation: the 16-dim `Observation::to_vec` from the bridge; extra
  normalization here: feature 2 (ln oracle) is centered at ln(1000) and
  scaled x10, everything else is already ~unit scale. No future info: the
  bridge builds the observation before the step executes.
- network: MLP 16-64-64-8 (ReLU), Adam 1e-3, Huber loss, gamma = 1.0
  (finite horizon; remaining_time is in the state).
- replay buffer 200k, batch 64, one gradient step per env step after warmup,
  target network sync every 2000 steps.
- epsilon 1.0 -> 0.05 linearly over the first 60% of training episodes.
- rewards scaled x100 for optimization only (recorded; evaluation uses the
  environment's own summary numbers, never the scaled reward).
- seeds: train 1,000,000+ep; validation 20,000-20,049 every 500 episodes
  (checkpoint selection); test 30,000-30,499; fresh 40,000-40,499.
- device: CPU, torch.manual_seed(7) for reproducibility.

Usage: python dqn_train.py [--episodes 12000] [--mode dynamic_duopoly]
       [--agent-order before|random|after] [--train-penalty 0.02] [--tag NAME]

M3R variants: --agent-order retrains under a different intra-step priority
(training AND evaluation use that ordering); --train-penalty shapes the
TRAINING reward only (evaluation always uses the standard 0.02 penalty);
--tag names the checkpoint/curve/results files (dqn_<tag>.*).
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn

from gym_env import AmmExecutionEnv

OUT = Path(__file__).resolve().parents[2] / "experiments/rl_execution/out"
LN_S0 = 6.907755278982137  # ln(1000)


def normalize(obs: list[float]) -> torch.Tensor:
    x = list(obs)
    x[2] = (x[2] - LN_S0) * 10.0
    return torch.tensor(x, dtype=torch.float32)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def reset_env(env: AmmExecutionEnv, seed: int, mode: str, **overrides):
    obs, _ = env.reset(seed=seed, mode=mode, **overrides)
    return obs


def greedy_episode(env: AmmExecutionEnv, net: QNet, seed: int, mode: str,
                   **overrides) -> dict:
    obs = reset_env(env, seed, mode, **overrides)
    done, info = False, {}
    while not done:
        with torch.no_grad():
            a = int(net(normalize(obs)).argmax())
        obs, _, done, _, info = env.step(a)
    return info["summary"]


def evaluate(env, net, seeds, mode, **overrides) -> float:
    return sum(greedy_episode(env, net, s, mode, **overrides)["shortfall_bps"]
               for s in seeds) / len(seeds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--mode", default="dynamic_duopoly")
    ap.add_argument("--agent-order", default="before",
                    choices=["before", "random", "after"])
    ap.add_argument("--train-penalty", type=float, default=0.02)
    ap.add_argument("--tag", default="dynamic_duopoly")
    args = ap.parse_args()

    torch.manual_seed(7)
    random.seed(7)
    torch.set_num_threads(1)
    # training-time overrides; evaluation uses the standard penalty
    train_ov = {"agent_order": args.agent_order,
                "unfinished_penalty": args.train_penalty}
    eval_ov = {"agent_order": args.agent_order}
    env = AmmExecutionEnv(mode=args.mode)
    net = QNet(env.obs_dim, env.n_actions)
    target = QNet(env.obs_dim, env.n_actions)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    buffer: deque = deque(maxlen=200_000)
    batch, warmup, sync_every = 64, 5_000, 2_000
    reward_scale = 100.0
    step_count = 0
    curve = []
    best = (float("inf"), None)
    val_seeds = list(range(20_000, 20_000 + args.n_val))

    t0 = time.time()
    for ep in range(args.episodes):
        eps = max(0.05, 1.0 - 0.95 * ep / (0.6 * args.episodes))
        obs = reset_env(env, 1_000_000 + ep, args.mode, **train_ov)
        state = normalize(obs)
        done = False
        while not done:
            if random.random() < eps:
                a = random.randrange(env.n_actions)
            else:
                with torch.no_grad():
                    a = int(net(state).argmax())
            obs, r, done, _, _ = env.step(a)
            nstate = normalize(obs)
            buffer.append((state, a, r * reward_scale, nstate, done))
            state = nstate
            step_count += 1

            if len(buffer) >= warmup:
                sample = random.sample(buffer, batch)
                s = torch.stack([x[0] for x in sample])
                acts = torch.tensor([x[1] for x in sample])
                rews = torch.tensor([x[2] for x in sample], dtype=torch.float32)
                s2 = torch.stack([x[3] for x in sample])
                dones = torch.tensor([float(x[4]) for x in sample])
                q = net(s).gather(1, acts.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    tgt = rews + (1.0 - dones) * target(s2).max(1).values
                loss = loss_fn(q, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if step_count % sync_every == 0:
                target.load_state_dict(net.state_dict())

        if (ep + 1) % args.eval_every == 0:
            val = evaluate(env, net, val_seeds, args.mode, **eval_ov)
            curve.append((ep + 1, val, eps))
            marker = ""
            if val < best[0]:
                best = (val, {k: v.clone() for k, v in net.state_dict().items()})
                marker = "  <- best"
            print(f"ep {ep+1:>6}  eps {eps:.2f}  val IS {val:8.2f} bps"
                  f"  ({time.time()-t0:.0f}s){marker}", flush=True)

    net.load_state_dict(best[1])
    torch.save(net.state_dict(), OUT / f"dqn_{args.tag}.pt")
    with open(OUT / f"m3_dqn_curve_{args.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "val_shortfall_bps", "eps"])
        w.writerows(curve)

    # held-out evaluation (standard penalty, training's agent order)
    with open(OUT / f"m3_dqn_results_{args.tag}.csv", "w", newline="") as f:
        w = None
        for label, base in [("test", 30_000), ("fresh", 40_000)]:
            for seed in range(base, base + args.n_test):
                s = greedy_episode(env, net, seed, args.mode, **eval_ov)
                s = {"policy": f"dqn_{args.tag}", "seed_set": label, **s}
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(s))
                    w.writeheader()
                w.writerow(s)
    print("selected checkpoint val IS:", round(best[0], 2))
    env.close()


if __name__ == "__main__":
    main()
