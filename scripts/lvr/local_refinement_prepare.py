#!/usr/bin/env python3
"""Freeze the Round-34 P2 local-refinement policy manifest.

This script reads only frozen training aggregates. It forms the center union
specified in minimal_v1_priority1_spec.md and emits only policies that are new
relative to the original 12-by-7 gap grid.
"""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr/workspace"
ALIGNMENT = LVR / "m3_loss_alignment.csv"
FRONTIER = LVR / "m3_amended_training_frontier.csv"
SELECTION = LVR / "m3_amended_training_selection.json"
CALIBRATION = LVR / "calibration_54_manifest.json"
SPEC = LVR / "minimal_v1_priority1_spec.md"
OUT = LVR / "m3_local_refinement_policies.json"

DIALS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 7.0, 10.0, 15.0, 25.0, 40.0)
ALPHAS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(value: float, grid: tuple[float, ...]) -> int:
    matches = [i for i, candidate in enumerate(grid) if math.isclose(value, candidate, abs_tol=1e-12)]
    assert len(matches) == 1, (value, grid)
    return matches[0]


def local_dials(center: float) -> list[float]:
    index = locate(center, DIALS)
    values = {center}
    if index > 0:
        values.add(math.sqrt(DIALS[index - 1] * center))
    if index + 1 < len(DIALS):
        values.add(math.sqrt(center * DIALS[index + 1]))
    return sorted(values)


def local_alphas(center: float) -> list[float]:
    index = locate(center, ALPHAS)
    values = {center}
    if index > 0:
        values.add((ALPHAS[index - 1] + center) / 2.0)
    if index + 1 < len(ALPHAS):
        values.add((center + ALPHAS[index + 1]) / 2.0)
    return sorted(values)


def number_id(value: float) -> str:
    return format(value, ".12g").replace(".", "p")


centers: dict[int, dict[tuple[float, float], set[str]]] = defaultdict(lambda: defaultdict(set))


def add_center(cell_idx: int, dial: float, alpha: float, role: str) -> None:
    dial = DIALS[locate(dial, DIALS)]
    alpha = ALPHAS[locate(alpha, ALPHAS)]
    centers[cell_idx][(dial, alpha)].add(role)


with ALIGNMENT.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 270
for row in rows:
    cell_idx = int(row["cell_idx"])
    rho = int(round(100 * float(row["rho"])))
    for selector in ("a", "l", "u"):
        add_center(
            cell_idx,
            float(row[f"pi_{selector}_star_dial_mult"]),
            float(row[f"pi_{selector}_star_alpha"]),
            f"constrained_{selector}_rho{rho:02d}",
        )

with FRONTIER.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 270
for row in rows:
    cell_idx = int(row["cell_idx"])
    rho = int(round(100 * float(row["rho"])))
    add_center(cell_idx, float(row["static_dial"]), 0.0, f"static_frontier_rho{rho:02d}")
    add_center(
        cell_idx,
        float(row["gap_dial"]),
        float(row["gap_alpha"]),
        f"gap_frontier_rho{rho:02d}",
    )

selection = json.loads(SELECTION.read_text())
assert selection["seed_block"] == {
    "start_inclusive": 20000,
    "end_exclusive": 20100,
    "n": 100,
}
for candidate in selection["matched_candidates"]:
    cell_idx = int(candidate["cell"]["cell_idx"])
    candidate_id = candidate["candidate_id"]
    for member_name in ("policy_1_lower_A", "policy_2"):
        policy = candidate[member_name]
        add_center(
            cell_idx,
            float(policy["dial_mult"]),
            float(policy["alpha"]),
            f"{candidate_id}_{member_name}",
        )

calibration = json.loads(CALIBRATION.read_text())
assert len(calibration["cells"]) == 54
cell_fees = {i: float(cell["fee"]) for i, cell in enumerate(calibration["cells"])}

original = {(dial, alpha) for dial in DIALS for alpha in ALPHAS}
cell_entries = []
total_centers = 0
total_new = 0
for cell_idx in range(54):
    assert centers[cell_idx], cell_idx
    candidates: dict[tuple[float, float], set[str]] = defaultdict(set)
    center_rows = []
    for (dial, alpha), roles in sorted(centers[cell_idx].items(), key=lambda item: (item[0][1], item[0][0])):
        total_centers += 1
        center_rows.append(
            {
                "dial_mult": dial,
                "alpha": alpha,
                "roles": sorted(roles),
            }
        )
        for refined_dial in local_dials(dial):
            for refined_alpha in local_alphas(alpha):
                candidates[(refined_dial, refined_alpha)].update(roles)

    new_policies = []
    for (dial, alpha), roles in sorted(candidates.items(), key=lambda item: (item[0][1], item[0][0])):
        if any(
            math.isclose(dial, coarse_dial, abs_tol=1e-12)
            and math.isclose(alpha, coarse_alpha, abs_tol=1e-12)
            for coarse_dial, coarse_alpha in original
        ):
            continue
        policy_id = f"refine_c{cell_idx:03d}_d{number_id(dial)}_a{number_id(alpha)}"
        new_policies.append(
            {
                "policy_id": policy_id,
                "dial_mult": dial,
                "f0": cell_fees[cell_idx] * dial,
                "alpha": alpha,
                "fee_cap": 0.30,
                "source_roles": sorted(roles),
            }
        )
    assert new_policies, cell_idx
    total_new += len(new_policies)
    cell_entries.append(
        {
            "cell_idx": cell_idx,
            "stratum_fee": cell_fees[cell_idx],
            "centers": center_rows,
            "new_policies": new_policies,
        }
    )

payload = {
    "schema_version": "m3-local-refinement-policies-v1",
    "phase": "Round 34 Phase C / P2",
    "source_stage": "frozen training aggregates only",
    "seed_block": {"start_inclusive": 20000, "end_exclusive": 20100, "n": 100},
    "original_grid": {"dial_mults": DIALS, "alphas": ALPHAS, "fee_cap": 0.30},
    "neighborhood": {
        "dial": "center plus geometric midpoints to existing adjacent coarse dials",
        "alpha": "center plus arithmetic midpoints to existing adjacent coarse alphas",
        "cartesian_product": True,
        "extrapolation": False,
        "coarse_members_rerun": False,
    },
    "counts": {
        "cells": 54,
        "distinct_centers_across_cells": total_centers,
        "new_cell_policies": total_new,
    },
    "input_sha256": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (ALIGNMENT, FRONTIER, SELECTION, CALIBRATION, SPEC, Path(__file__).resolve())
    },
    "cells": cell_entries,
}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
if OUT.exists() and OUT.read_bytes() != encoded:
    raise RuntimeError(f"refusing to overwrite different frozen manifest: {OUT}")
OUT.write_bytes(encoded)
print(f"cells=54 centers={total_centers} new_cell_policies={total_new}")
print(f"wrote {OUT}")
