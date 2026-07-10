"""Gymnasium-style Python wrapper around the Rust execution environment.

Talks JSON-lines to `rl_equilibrium_bridge` over stdin/stdout. No hard dependency on
gymnasium: the class duck-types reset()/step() so it works with most RL
libraries (or plain Q-learning loops) directly. If gymnasium is installed,
`spaces` attributes are populated.

Usage:
    env = AmmExecutionEnv(mode="dynamic_duopoly")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "rl_equilibrium_bridge"

MODES = ("constant_duopoly", "dynamic_monopoly", "dynamic_duopoly")


class AmmExecutionEnv:
    def __init__(self, mode: str = "dynamic_duopoly", binary: Path | str = DEFAULT_BINARY,
                 sigma: float | None = None, arb_speed: float | None = None):
        assert mode in MODES, f"mode must be one of {MODES}"
        self.mode = mode
        self.sigma = sigma
        self.arb_speed = arb_speed
        binary = Path(binary)
        if not binary.exists():
            raise FileNotFoundError(
                f"{binary} not found - run `cargo build --release --bin rl_equilibrium_bridge` first")
        self._proc = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        hello = self._read()
        assert hello["type"] == "hello", hello
        self.n_actions: int = hello["n_actions"]
        self.obs_dim: int = hello["obs_dim"]
        try:  # optional gymnasium spaces
            import gymnasium as gym
            import numpy as np
            self.action_space = gym.spaces.Discrete(self.n_actions)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=float)
        except ImportError:
            pass

    def _send(self, msg: dict) -> None:
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def _read(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("environment process died")
        return json.loads(line)

    def reset(self, seed: int = 0, *, mode: str | None = None, **overrides):
        """Reset with optional per-episode overrides.

        Supported override keys mirror the bridge's reset command: sigma,
        arb_speed, gas, agent_order, noise_intensity_scale,
        unfinished_penalty, completion_rule, lp_regime, jit_regime.
        """
        req = {"cmd": "reset", "seed": seed, "mode": mode or self.mode}
        if self.sigma is not None:
            req.setdefault("sigma", self.sigma)
        if self.arb_speed is not None:
            req.setdefault("arb_speed", self.arb_speed)
        req.update(overrides)
        self._send(req)
        msg = self._read()
        assert msg["type"] == "state", msg
        return msg["obs"], {"raw": msg["raw"]}

    def step(self, action: int):
        self._send({"cmd": "step", "action": int(action)})
        msg = self._read()
        assert msg["type"] == "transition", msg
        info = {"raw": msg["raw"]}
        if "summary" in msg:
            info["summary"] = msg["summary"]
        return msg["obs"], msg["reward"], msg["done"], False, info

    def close(self):
        try:
            self._send({"cmd": "close"})
        except (BrokenPipeError, ValueError):
            pass
        self._proc.terminate()


if __name__ == "__main__":
    # Smoke test: run one TWAP-ish episode through the bridge.
    env = AmmExecutionEnv()
    obs, _ = env.reset(seed=42)
    total, done = 0.0, False
    while not done:
        obs, reward, done, _, info = env.step(7)  # always split
        total += reward
    print(f"episode reward: {total:.6f}")
    print(json.dumps(info["summary"], indent=2))
    env.close()
