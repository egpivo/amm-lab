"""Gymnasium-style wrapper around the Rust execution environment."""

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "rl_equilibrium_bridge"

MODES = ("constant_duopoly", "dynamic_monopoly", "dynamic_duopoly")


class BridgeProtocolError(RuntimeError):
    pass


class AmmExecutionEnv:
    def __init__(
        self,
        mode: str = "dynamic_duopoly",
        binary: Path | str = DEFAULT_BINARY,
        sigma: float | None = None,
        arb_speed: float | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.sigma = sigma
        self.arb_speed = arb_speed
        binary = Path(binary)
        if not binary.exists():
            raise FileNotFoundError(
                f"{binary} not found - run `cargo build --release --bin rl_equilibrium_bridge` first"
            )
        self._proc = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        hello = self._read()
        self._expect_type(hello, "hello")
        self.n_actions = int(hello["n_actions"])
        self.obs_dim = int(hello["obs_dim"])
        try:
            import gymnasium as gym
            import numpy as np

            self.action_space = gym.spaces.Discrete(self.n_actions)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=float
            )
        except ImportError:
            pass

    def __enter__(self) -> "AmmExecutionEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _expect_type(message: Mapping[str, Any], expected: str) -> None:
        if message.get("type") != expected:
            raise BridgeProtocolError(f"expected {expected!r}, received {message!r}")

    def _send(self, message: Mapping[str, Any]) -> None:
        if self._proc.stdin is None or self._proc.poll() is not None:
            raise BridgeProtocolError("environment process is not running")
        self._proc.stdin.write(json.dumps(message) + "\n")
        self._proc.stdin.flush()

    def _read(self) -> dict[str, Any]:
        if self._proc.stdout is None:
            raise BridgeProtocolError("environment process has no stdout")
        line = self._proc.stdout.readline()
        if not line:
            raise BridgeProtocolError(
                f"environment process exited with code {self._proc.poll()}"
            )
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            raise BridgeProtocolError(f"invalid bridge response: {line!r}") from error
        if not isinstance(message, dict):
            raise BridgeProtocolError(f"bridge response must be an object: {message!r}")
        return message

    def reset(
        self, seed: int = 0, *, mode: str | None = None, **overrides: Any
    ) -> tuple[list[float], dict[str, Any]]:
        req = {"cmd": "reset", "seed": seed, "mode": mode or self.mode}
        if self.sigma is not None:
            req.setdefault("sigma", self.sigma)
        if self.arb_speed is not None:
            req.setdefault("arb_speed", self.arb_speed)
        req.update(overrides)
        self._send(req)
        msg = self._read()
        self._expect_type(msg, "state")
        return msg["obs"], {"raw": msg["raw"]}

    def step(
        self, action: int
    ) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        self._send({"cmd": "step", "action": int(action)})
        msg = self._read()
        self._expect_type(msg, "transition")
        info = {"raw": msg["raw"]}
        if "summary" in msg:
            info["summary"] = msg["summary"]
        return msg["obs"], msg["reward"], msg["done"], False, info

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            self._send({"cmd": "close"})
        except (BridgeProtocolError, BrokenPipeError, ValueError):
            pass
        try:
            self._proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()


if __name__ == "__main__":
    with AmmExecutionEnv() as env:
        observation, _ = env.reset(seed=42)
        total, done = 0.0, False
        while not done:
            observation, reward, done, _, info = env.step(7)
            total += reward
        print(f"episode reward: {total:.6f}")
        print(json.dumps(info["summary"], indent=2))
