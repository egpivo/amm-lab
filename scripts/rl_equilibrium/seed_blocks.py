"""Seed-block registry for VTR-1 (Gate 1). Pure data and arithmetic: no simulator import.

Every block is a finite half-open interval [start, end) with one declared use. A run
must ask for a range contained in a registered block of its use; a range that
overlaps a block of another use, or that leaves every block, aborts before any
simulation. Protected uses (pilot, final, w0) additionally need an authorization file.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Block:
    name: str
    start: int
    end: int  # exclusive
    use: str


# Historical blocks follow the committed generators (12,000 training episodes; n_val up to 200;
# n_test/n_completion 500; verify_paper_artifacts FINAL_SEEDS; sensitivity LP/JIT blocks).
BLOCKS: tuple[Block, ...] = (
    Block("hist_train", 1_000_000, 1_012_000, "train"),
    Block("dev_val", 20_000, 20_200, "val"),
    Block("dev_test", 30_000, 30_500, "eval"),
    Block("dev_fresh", 40_000, 40_500, "eval"),
    Block("dev_final_legacy", 90_000, 91_000, "final"),
    Block("dev_lp", 95_000, 95_500, "eval"),
    Block("dev_jit", 96_000, 96_500, "eval"),
    Block("vtr1_val_B", 150_000, 150_200, "vtr1_val"),
    Block("vtr1_pilot", 170_000, 170_200, "pilot"),
    Block("vtr1_final", 190_000, 191_000, "final"),
    Block("w0_paths_DB1", 200_000, 300_000, "w0"),
    Block("w0_days_DB2", 300_000, 358_000, "w0"),
    Block("w6_train_R0", 10_000_000, 10_012_000, "vtr1_train"),
    Block("w6_train_R1", 11_000_000, 11_012_000, "vtr1_train"),
    Block("w6_train_R2", 12_000_000, 12_012_000, "vtr1_train"),
    Block("pilot_train", 30_000_000, 30_012_000, "pilot_train"),
)

PROTECTED_USES = {"vtr1_train", "pilot_train", "vtr1_val", "pilot", "final", "w0"}


class SeedBlockError(SystemExit):
    pass


def _overlaps(start: int, end: int):
    return [b for b in BLOCKS if start < b.end and b.start < end]


def require(
    use: str,
    base: int,
    n: int,
    authorization: str | None = None,
    scope: str | None = None,
) -> Block:
    """Validate that seeds [base, base+n) belong to a registered block of `use`.

    A protected use needs an authorization JSON whose `scope` equals `scope` (default: `use`;
    e.g. a W6 training run authorizes both its training and its validation block under the
    scope "vtr1_train") and whose `blocks` names the block.
    """
    if n <= 0:
        raise SeedBlockError(f"seed count must be positive, got {n}")
    start, end = base, base + n
    hits = _overlaps(start, end)
    for b in hits:
        if b.use != use:
            raise SeedBlockError(
                f"seeds [{start},{end}) for use '{use}' overlap block {b.name} "
                f"[{b.start},{b.end}) reserved for '{b.use}'"
            )
    inside = [b for b in hits if b.use == use and b.start <= start and end <= b.end]
    if not inside:
        raise SeedBlockError(
            f"seeds [{start},{end}) for use '{use}' are not inside a registered '{use}' block"
        )
    block = inside[0]
    if use in PROTECTED_USES:
        _check_authorization(scope or use, block, authorization)
    return inside[0]


def _check_authorization(scope: str, block: Block, authorization: str | None) -> None:
    """A protected scope needs a JSON file with this scope and the block name in `blocks`."""
    if not (authorization and Path(authorization).is_file()):
        raise SeedBlockError(f"scope '{scope}' is protected: pass --authorization pointing to an existing file")
    try:
        record = json.loads(Path(authorization).read_text())
    except ValueError as error:
        raise SeedBlockError(f"authorization {authorization} is not JSON: {error}") from error
    if record.get("scope") != scope or block.name not in record.get("blocks", []):
        raise SeedBlockError(
            f"authorization {authorization} does not cover scope '{scope}' / block {block.name}"
        )


def assert_registry_disjoint() -> None:
    ordered = sorted(BLOCKS, key=lambda b: b.start)
    for a, b in zip(ordered, ordered[1:]):
        assert a.end <= b.start, f"blocks overlap: {a.name} / {b.name}"


def check_new_outputs(paths) -> None:
    """VTR-1 outputs must not exist and must not live in the archived data directory."""
    archived = (Path(__file__).resolve().parents[2] / "data/rl_equilibrium").resolve()
    for raw in paths:
        p = Path(raw)
        if p.exists():
            raise SeedBlockError(f"{p} exists; VTR-1 outputs are never overwritten")
        if archived in p.resolve().parents:
            raise SeedBlockError(f"{p} is inside the archived directory {archived}")
