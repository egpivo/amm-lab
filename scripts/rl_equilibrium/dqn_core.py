from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch import nn

from common import OUT, mean
from gym_env import AmmExecutionEnv


LN_S0 = 6.907755278982137
Summary = dict[str, Any]


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def normalize(observation: Sequence[float]) -> torch.Tensor:
    values = list(observation)
    values[2] = (values[2] - LN_S0) * 10.0
    return torch.tensor(values, dtype=torch.float32)


def reset_env(
    env: AmmExecutionEnv, seed: int, mode: str, **overrides: Any
) -> list[float]:
    observation, _ = env.reset(seed=seed, mode=mode, **overrides)
    return observation


def greedy_action(net: QNet, observation: Sequence[float]) -> int:
    with torch.inference_mode():
        return int(net(normalize(observation)).argmax().item())


def greedy_episode(
    env: AmmExecutionEnv,
    net: QNet,
    seed: int,
    mode: str,
    *,
    observation_transform: Callable[[list[float]], list[float]] | None = None,
    action_transform: Callable[[int], int] | None = None,
    **overrides: Any,
) -> Summary:
    observation = reset_env(env, seed, mode, **overrides)
    done = False
    info: dict[str, Any] = {}
    while not done:
        policy_observation = (
            observation_transform(observation) if observation_transform else observation
        )
        action = greedy_action(net, policy_observation)
        if action_transform:
            action = action_transform(action)
        observation, _, done, _, info = env.step(action)
    return info["summary"]


def evaluate(
    env: AmmExecutionEnv,
    net: QNet,
    seeds: Iterable[int],
    mode: str,
    **overrides: Any,
) -> float:
    return mean(
        greedy_episode(env, net, seed, mode, **overrides)["shortfall_bps"]
        for seed in seeds
    )


def load_q_network(env: AmmExecutionEnv, tag: str) -> QNet:
    network = QNet(env.obs_dim, env.n_actions)
    checkpoint = torch.load(
        OUT / f"dqn_{tag}.pt", map_location="cpu", weights_only=True
    )
    network.load_state_dict(checkpoint)
    network.eval()
    return network
