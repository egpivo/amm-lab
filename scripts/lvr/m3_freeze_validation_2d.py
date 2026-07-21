#!/usr/bin/env python3
"""Freeze the amended M3 candidate universe before fresh held-out simulation."""

import hashlib
import json
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr"
SELECTION = LVR / "m3_amended_training_selection.json"
SELECTION_HASH = LVR / "m3_amended_training_selection.sha256"
PLAN = LVR / "m3_amended_validation_plan.json"
PLAN_HASH = LVR / "m3_amended_validation_plan.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()


selection_hash = sha256(SELECTION)
assert SELECTION_HASH.read_text().split()[0] == selection_hash
selection = json.loads(SELECTION.read_text())
assert selection["seed_block"] == {
    "start_inclusive": 20000,
    "end_exclusive": 20100,
    "n": 100,
}
assert len(selection["matched_candidates"]) == 106

frontier_pairs = []
for cell in selection["cells"]:
    for target in cell["targets"]:
        rho = target["rho"]
        frontier_pairs.append(
            {
                "frontier_id": (
                    f"amended_frontier_c{cell['cell']['cell_idx']:03d}_"
                    f"rho{int(round(100 * rho)):02d}"
                ),
                "cell": cell["cell"],
                "empirical_support": cell["empirical_support"],
                "rho": rho,
                "training_S0": cell["S0_training"],
                "target_s_training": target["target_s_training"],
                "static_policy": target["static_frontier"],
                "gap_policy": target["gap_frontier"],
                "strict_gap_improvement_on_training": target[
                    "strict_gap_frontier_improvement"
                ],
                "training_delta_U_gap_minus_static": target[
                    "training_delta_U_gap_minus_static"
                ],
            }
        )
assert len(frontier_pairs) == 270

inputs = [
    LVR / "calibration_54_manifest.json",
    LVR / "m3_policy_support_audit.json",
    LVR / "m3_policy_support_audit.sha256",
    LVR / "m3_joint_support.json",
    LVR / "m3_joint_support.csv",
    SELECTION,
    SELECTION_HASH,
    LVR / "m3_amended_training_frontier.csv",
    LVR / "m3_amended_diagnostics.json",
    LVR / "m3_amended_diagnostics.sha256",
    ROOT / "scripts/lvr/m3_select_2d.py",
    ROOT / "scripts/lvr/m3_diagnostics_analyze.py",
    ROOT / "scripts/lvr/m3_validate_select_2d.py",
    ROOT / "src/bin/lvr_m3_train_2d.rs",
    ROOT / "src/bin/lvr_m3_validate_2d.rs",
    ROOT / "src/campbell/fee_policy.rs",
    ROOT / "src/campbell/simulation.rs",
    Path(__file__).resolve(),
    *(LVR / f"m3_amended_training_rows_shard{i}.csv.gz" for i in range(6)),
    *(LVR / f"m3_amended_diagnostics_shard{i}.json.gz" for i in range(6)),
]
assert all(path.exists() for path in inputs)

plan = {
    "step": "M3 amended immutable fresh-validation plan",
    "source_stage": "amended two-dimensional family; training-only selection and diagnostics",
    "chronology": {
        "disclosure": (
            "The narrow-family held-out runs predated the design audit and are superseded. "
            "Their seeds are permanently retired and are not reused here."
        ),
        "retired_seed_blocks": [
            {"role": "narrow validation", "start_inclusive": 30000, "end_exclusive": 30200},
            {"role": "narrow final", "start_inclusive": 90000, "end_exclusive": 90400},
        ],
    },
    "training_seed_block": selection["seed_block"],
    "validation_seed_block": {"start_inclusive": 40000, "end_exclusive": 40200, "n": 200},
    "reserved_final_seed_block": {"start_inclusive": 91000, "end_exclusive": 91400, "n": 400},
    "policy_family": {
        "formula": "clip(f0 + alpha * abs(stale_gap_bps) / 10000, f0, 0.30)",
        "f0_dial_mults": selection["grid"]["dial_mults"],
        "alphas": selection["grid"]["alphas"],
        "fee_cap": selection["grid"]["fee_cap"],
        "static_boundary": "alpha=0 is simulated as the identical fixed-fee policy",
    },
    "matched_service_rule": {
        "candidate_universe": "all 106 training-qualified opposite-orientation static/gap pairs",
        "support": (
            "validation pair mismatch <= 0.05 training S0 and each policy target gap "
            "<= 0.10 training S0"
        ),
        "direction": "mean delta A < 0 and mean delta U < 0 for policy_1_lower_A minus policy_2",
        "inference": "both one-sided paired 95% CI upper bounds < 0",
        "winner_rule": (
            "maximize min(-t_delta_A,-t_delta_U), then lower validation service mismatch, "
            "then candidate_id"
        ),
    },
    "frontier_rule": {
        "universe": "270 training-frozen static versus full-gap-family frontier pairs",
        "validation": "no re-optimization on validation seeds",
        "evidence_gate": (
            "both frozen policies retain mean S >= rho*training S0 and the one-sided paired "
            "95% lower bound for delta U (gap minus static) is > 0"
        ),
    },
    "matched_candidates": selection["matched_candidates"],
    "frontier_pairs": frontier_pairs,
    "input_artifacts_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
}

payload = json_bytes(plan)
if PLAN.exists() and PLAN.read_bytes() != payload:
    raise RuntimeError(f"refusing to overwrite frozen validation plan: {PLAN}")
if not PLAN.exists():
    PLAN.write_bytes(payload)
digest = hashlib.sha256(payload).hexdigest()
PLAN_HASH.write_text(f"{digest}  {PLAN.name}\n")
print(f"matched_candidates={len(plan['matched_candidates'])}")
print(f"frontier_pairs={len(plan['frontier_pairs'])}")
print(f"validation_seed_block={plan['validation_seed_block']}")
print(f"reserved_final_seed_block={plan['reserved_final_seed_block']}")
print(f"plan_sha256={digest}")
