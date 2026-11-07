"""VTR-1 scenario file (JSON): one specification consumed by training, validation, kappa
tuning, the reference policies and DQN evaluation. Rust (`ScenarioConfig`) parses the same file.

    {"name": "S1-0.65", "sigma": 0.65}                                  # Group 1
    {"name": "V-30", "path_spec": {"Scenario": {"sigma_bar": 0.5, ...}}}  # Groups 2-5

The Python side forwards `sigma` and `path_spec` to the bridge reset message; unknown keys
are rejected here and malformed `path_spec` values are rejected by the bridge (never ignored).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

ALLOWED = {"name", "sigma", "path_spec"}


def load(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - ALLOWED:
        raise SystemExit(f"scenario file {path}: expected an object with keys in {sorted(ALLOWED)}")
    if "sigma" in raw and not (isinstance(raw["sigma"], (int, float)) and raw["sigma"] > 0):
        raise SystemExit(f"scenario file {path}: sigma must be a positive number")
    return raw


def overrides(path: str | None) -> dict[str, Any]:
    """Keyword overrides for `AmmExecutionEnv.reset`."""
    raw = load(path)
    return {k: v for k, v in raw.items() if k in ("sigma", "path_spec")}


def file_sha256(path: str | None) -> str | None:
    return None if path is None else hashlib.sha256(Path(path).read_bytes()).hexdigest()


# The five (mode, ordering) cells for which the reference policies and kappa tuning exist.
CELLS = (
    "dynamic_duopoly/before",
    "dynamic_duopoly/random",
    "dynamic_duopoly/after",
    "constant_duopoly/before",
    "dynamic_monopoly/before",
)


def check_cell(cell: str) -> tuple[str, str]:
    if cell not in CELLS:
        raise SystemExit(f"cell must be one of {CELLS}, got {cell!r}")
    mode, order = cell.split("/")
    return mode, order
