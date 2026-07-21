#!/usr/bin/env python3
"""Write the immutable Round-34 P2 plan with input hashes."""

import hashlib
import json
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr/workspace"
OUT = LVR / "m3_local_refinement_plan_v3.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


policy_manifest = json.loads((LVR / "m3_local_refinement_policies.json").read_text())
assert policy_manifest["counts"]["cells"] == 54
assert policy_manifest["counts"]["new_cell_policies"] == 3468

inputs = [
    Path("Cargo.toml"),
    Path("Cargo.lock"),
    Path("src/bin/lvr_refine.rs"),
    Path("src/campbell/fee_policy.rs"),
    Path("src/campbell/gbm.rs"),
    Path("src/campbell/simulation.rs"),
    Path("src/campbell/summary.rs"),
    Path("scripts/lvr/local_refinement_prepare.py"),
    Path("scripts/lvr/local_refinement_plan.py"),
    Path("scripts/lvr/local_refinement_analyze.py"),
    Path(".local/lvr/m3_local_refinement_policies.json"),
    Path(".local/lvr/m3_loss_alignment.csv"),
    Path(".local/lvr/m3_amended_training_frontier.csv"),
    Path(".local/lvr/m3_amended_training_selection.json"),
    Path(".local/lvr/m3_joint_support.json"),
    Path(".local/lvr/calibration_54_manifest.json"),
    Path(".local/lvr/minimal_v1_priority1_spec.md"),
    Path(".local/lvr/reviewer_directive_round34.md"),
    Path(".local/lvr/current_freeze_hash_manifest.sha256"),
    Path(".local/lvr/m3_local_refinement_plan.json"),
    Path(".local/lvr/m3_local_refinement_plan_v2_unexecuted.json"),
    *[Path(f".local/lvr/m3_amended_training_rows_shard{i}.csv.gz") for i in range(6)],
]
for relative in inputs:
    assert (ROOT / relative).is_file(), relative

plan = {
    "schema_version": "m3-local-refinement-plan-v3",
    "created_at": "2026-07-16",
    "status_before_run": "FROZEN",
    "phase": "Round 34 Phase C / P2",
    "purpose": "Training-only local finite refinement around frozen selector centers",
    "supersedes": {
        "plan": ".local/lvr/m3_local_refinement_plan.json",
        "plan_sha256": "c04e4b81f822ec758b530a43f4da47c1ae518663e681011eeb8e6e6d72f4c8f7",
        "reason": "The first execution was stopped before analysis because full step-record retention implied multi-hour runtime. Partial shards were archived and are prohibited analysis inputs.",
        "scientific_spec_changed": False,
    },
    "pre_run_plan_supersession": {
        "plan": ".local/lvr/m3_local_refinement_plan_v2_unexecuted.json",
        "plan_sha256": "4e3a2cda4f1a71e08500a037522bd56f465bbfc8ff04336b07a826ea3dfb7f30",
        "reason": "A pre-run manifest check found workspace formatting had changed the P1 runner source hash. No v2 runner was launched; the baseline was corrected before this plan was written.",
        "scientific_spec_changed": False,
    },
    "seed_domain": {
        "start_inclusive": 20000,
        "end_exclusive": 20100,
        "count_per_new_cell_policy": 100,
        "classification": "training only",
    },
    "configuration": {
        "cells": 54,
        "horizon_days": 7,
        "n_steps": 604800,
        "dt_seconds": 1,
        "policy_lag_steps": 300,
        "outside_venue_cost": 0.001,
        "fee_cap": 0.30,
        "initial_pool_value": 40000000.0,
        "original_gap_policies_per_cell": 84,
        "distinct_centers_across_cells": policy_manifest["counts"]["distinct_centers_across_cells"],
        "new_cell_policies": policy_manifest["counts"]["new_cell_policies"],
        "expected_new_rows": policy_manifest["counts"]["new_cell_policies"] * 100,
        "coarse_members_rerun": False,
        "record_retention": "primitive event ledger plus fee-bearing and terminal step records only",
        "record_retention_equivalence_test": "campbell::simulation::tests::compact_event_run_preserves_event_summary",
        "refined_set": "frozen coarse 84-member grid union new local midpoint policies",
        "bounds": "no dial or alpha extrapolation beyond the original grid",
    },
    "selectors": {
        "service_targets_rho": [0.2, 0.4, 0.6, 0.8, 0.95],
        "objectives": ["constrained A", "constrained L", "constrained U"],
        "frontiers": ["static alpha=0", "amended gap including alpha=0 boundary"],
        "tie_break": "primary objective; lower alpha; lower dial_mult; stable policy_id",
        "baseline_counts_required": {
            "A_vs_U": "158/270; supported 79/135",
            "L_vs_U": "196/270; supported 90/135",
            "A_vs_L": "100/270; supported 43/135",
            "strict_gap_frontier": "249/270",
        },
    },
    "acceptance": {
        "integrity": "100 rows per new cell-policy; L=A-B and U=fees-L within 1e-10 relative tolerance",
        "stable_l_u_supported_min_share": 0.50,
        "value_tolerance_absolute": 800.0,
        "value_stable_supported_min_share": 0.80,
        "selector_sensitive_supported_u_change_share_strictly_above": 0.20,
        "all_counts_reported": True,
        "failure_action": "stop P3-P4 and report classification",
    },
    "penalty_reanalysis": {
        "recompute_lambda_star": True,
        "recompute_complete_penalized_surplus_envelope": True,
        "normalization": "lambda_star*S0/V0",
        "penalized_L_large_penalty_limit": True,
        "exact_surplus_recovery_required": "270/270",
        "reuse_coarse_thresholds": False,
    },
    "outputs": [
        *[f".local/lvr/m3_local_refinement_rows_shard{i}.csv.gz" for i in range(6)],
        ".local/lvr/m3_local_refinement_alignment.csv",
        ".local/lvr/m3_local_refinement_frontier.csv",
        ".local/lvr/m3_local_refinement_selector_changes.csv",
        ".local/lvr/m3_local_refinement_penalties.csv",
        ".local/lvr/m3_local_refinement_penalty_breakpoints.csv",
        ".local/lvr/m3_local_refinement_report.md",
    ],
    "input_sha256": {str(relative): sha256(ROOT / relative) for relative in inputs},
    "prohibited_inputs": {
        "validation_seed_min": 40000,
        "final_seed_min": 91000,
        "validation_or_final_rows": "must not be read",
        "new_held_out_block": False,
        "selector_promotion": False,
    },
    "stop_gate": "After P2, stop for reviewer review; P3 and P4 remain blocked.",
}
payload = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
if OUT.exists() and OUT.read_bytes() != payload:
    raise RuntimeError(f"refusing to overwrite different frozen plan: {OUT}")
OUT.write_bytes(payload)
print(f"plan_sha256={hashlib.sha256(payload).hexdigest()}")
print(f"wrote {OUT}")
