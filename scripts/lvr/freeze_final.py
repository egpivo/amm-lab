#!/usr/bin/env python3
"""Freeze the single validation-grid final-block candidate selected on validation."""

import hashlib
import json
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr/workspace"
VALIDATION_PLAN = LVR / "m3_validation_plan.json"
VALIDATION_PLAN_HASH = LVR / "m3_validation_plan.sha256"
VALIDATION_SELECTION = LVR / "m3_validation_selection.json"
FINAL_PLAN = LVR / "m3_final_plan.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()


validation_plan_hash = sha256(VALIDATION_PLAN)
assert VALIDATION_PLAN_HASH.read_text().split()[0] == validation_plan_hash
validation_plan = json.loads(VALIDATION_PLAN.read_text())
selection = json.loads(VALIDATION_SELECTION.read_text())
assert selection["validation_plan_sha256"] == validation_plan_hash
winner_id = selection["matched_summary"]["winner_candidate_id"]
assert winner_id is not None
assert selection["winner"]["both_one_sided_95_upper_below_zero"]
candidate = next(
    x for x in validation_plan["matched_candidates"] if x["candidate_id"] == winner_id
)

inputs = [
    VALIDATION_PLAN,
    VALIDATION_PLAN_HASH,
    VALIDATION_SELECTION,
    *(LVR / f"m3_validation_rows_shard{i}.csv.gz" for i in range(6)),
    ROOT / "scripts/lvr/validate_select.py",
    Path(__file__).resolve(),
]
plan = {
    "step": "final-block immutable final-block plan",
    "source_stage": "validation-selected; no final-seed search",
    "final_seed_block": validation_plan["final_seed_block"],
    "candidate": candidate,
    "validation_selection_snapshot": {
        "candidate_id": winner_id,
        "studentized_margin_score": selection["winner"]["studentized_margin_score"],
        "service_mismatch_over_training_S0": selection["winner"][
            "service_mismatch_over_training_S0"
        ],
        "delta_a": selection["winner"]["delta_a_policy_1_minus_policy_2"],
        "delta_u": selection["winner"]["delta_u_policy_1_minus_policy_2"],
    },
    "verification_rule": {
        "service": "pair mismatch <= 0.05 training S0 and each policy target gap <= 0.10 training S0",
        "direction": "mean delta A < 0 and mean delta U < 0 for policy_1_lower_A minus policy_2",
        "inference": "both one-sided paired 95% CI upper bounds < 0",
        "selection": "none; exactly one validation-selected candidate is evaluated",
    },
    "input_artifacts_sha256": {
        str(path.relative_to(ROOT)): sha256(path) for path in inputs
    },
}
payload = json_bytes(plan)
if FINAL_PLAN.exists() and FINAL_PLAN.read_bytes() != payload:
    raise RuntimeError(f"refusing to overwrite frozen final plan: {FINAL_PLAN}")
if not FINAL_PLAN.exists():
    FINAL_PLAN.write_bytes(payload)
digest = hashlib.sha256(payload).hexdigest()
FINAL_PLAN.with_suffix(".sha256").write_text(f"{digest}  {FINAL_PLAN.name}\n")
print(f"candidate={winner_id}")
print(f"final_seed_block={plan['final_seed_block']}")
print(f"final_plan_sha256={digest}")
