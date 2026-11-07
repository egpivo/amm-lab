"""VTR-1 scenario file (JSON): one specification consumed by training, validation, kappa
tuning, the reference policies and DQN evaluation. Rust (`ScenarioConfig`) parses the same file.

    {"name": "S1-0.65", "sigma": 0.65}                                  # Group 1
    {"name": "V-30", "path_spec": {"Scenario": {"sigma_bar": 0.5, ...}}}  # Groups 2-5

The Python side forwards `sigma` and `path_spec` to the bridge reset message; unknown keys
are rejected here and malformed `path_spec` values are rejected by the bridge (never ignored).
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ALLOWED = {"name", "sigma", "path_spec"}


def _num(name: str, x: Any, *, positive: bool = True, gt: float | None = None) -> float:
    """A JSON number (never a bool), finite, and > 0 (or >= 0 / > gt as stated)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise SystemExit(f"scenario: {name} must be a finite number, got {x!r}")
    if gt is not None and not x > gt:
        raise SystemExit(f"scenario: {name} must be > {gt}, got {x}")
    if gt is None and positive and not x > 0:
        raise SystemExit(f"scenario: {name} must be > 0, got {x}")
    if gt is None and not positive and x < 0:
        raise SystemExit(f"scenario: {name} must be >= 0, got {x}")
    return float(x)


def _fields(name: str, obj: Any, required: set[str], optional: set[str]) -> dict:
    if not isinstance(obj, dict):
        raise SystemExit(f"scenario: {name} must be an object")
    unknown = set(obj) - required - optional
    missing = required - set(obj)
    if unknown or missing:
        raise SystemExit(f"scenario: {name} unknown keys {sorted(unknown)} / missing {sorted(missing)}")
    return obj


def validate_path_spec(spec: Any) -> None:
    """Mirror of Rust `PathSpec` + `PathSpec::validate` (serde externally tagged enum)."""
    if spec == "Baseline":
        return
    if not (isinstance(spec, dict) and set(spec) == {"Scenario"}):
        raise SystemExit(f"scenario: path_spec must be \"Baseline\" or {{\"Scenario\": ...}}, got {spec!r}")
    body = _fields("Scenario", spec["Scenario"], {"sigma_bar"}, {"vol", "t"})
    _num("sigma_bar", body["sigma_bar"])
    if body.get("vol") is not None:
        vol = _fields("vol", body["vol"], {"s", "k"}, {"tau"})
        _num("s", vol["s"], positive=False)
        _num("k", vol["k"])
        if vol.get("tau") is not None:
            _num("tau", vol["tau"])
    if body.get("t") is not None:
        t = _fields("t", body["t"], {"nu", "cap"}, set())
        _num("nu", t["nu"], gt=2.0)
        _num("cap", t["cap"])


def _reject_constant(token: str):
    raise SystemExit(f"scenario: non-finite JSON token {token}")


def load(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        raw = json.loads(Path(path).read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError) as error:
        raise SystemExit(f"scenario file {path}: {error}") from error
    return validate(raw)


def validate(raw: Any) -> dict[str, Any]:
    raw = _fields("scenario", raw, set(), ALLOWED)
    if "name" in raw and not isinstance(raw["name"], str):
        raise SystemExit("scenario: name must be a string")
    if "sigma" in raw:
        _num("sigma", raw["sigma"])
    if "path_spec" in raw:
        validate_path_spec(raw["path_spec"])
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
