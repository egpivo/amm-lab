#!/usr/bin/env python3
"""Freeze the validation-grid narrow-policy-family finding and design amendment."""

import hashlib
import json
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr/workspace"
TRAIN = ROOT / "src/bin/lvr_train.rs"
SELECT = ROOT / "scripts/lvr/select.py"
POLICY = ROOT / "src/campbell/fee_policy.rs"
AMENDED_TRAIN = ROOT / "src/bin/lvr_train_2d.rs"
OUT = LVR / "m3_policy_support_audit.json"
REPORT = LVR / "m3_policy_support_audit.md"
DIALS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 7.0, 10.0, 15.0, 25.0, 40.0]
ALPHAS_AMENDED = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


train_source = TRAIN.read_text()
select_source = SELECT.read_text()
policy_source = POLICY.read_text()
amended_source = AMENDED_TRAIN.read_text()
source_assertions = {
    "primary_clock_one_second": "dt_hours: 1.0 / 3600.0" in train_source,
    "episode_one_week": "n_steps: 604_800" in train_source,
    "signal_staleness_300_steps": "policy_lag: 300" in train_source,
    "lookback_20_physical_hours": "lookback_hours: 20.0" in train_source,
    "poisson_primitive_arrivals": "arrival_model: ArrivalModel::Poisson" in train_source,
    "cell_hazards_from_manifest": (
        "calibration_54_manifest.json" in train_source
        and "lambda_arb_star_SOLVED" in train_source
        and "lambda_fund_star_SOLVED" in train_source
    ),
    "frontier_uses_true_service_constraint": 'if means[(ci, "static", dm)]["s"] >= target'
    in select_source,
    "nearest_service_is_separate": "min(curve, key=lambda c: abs(c[1] - target))"
    in select_source,
    "gap_formula_is_linear_absolute_gap": (
        "self.base_fee + self.gap_multiplier * obs.oracle_gap_bps.abs() / 10_000.0"
        in policy_source
    ),
}
assert all(source_assertions.values()), source_assertions
amended_source_assertions = {
    "fixed_cap_30pct": "const FEE_CAP: f64 = 0.30" in amended_source,
    "positive_alpha_grid": "[0.05, 0.10, 0.25, 0.50, 1.0, 2.0]" in amended_source,
    "alpha_zero_written_as_static_alias": '"gap",' in amended_source
    and "alpha_zero_static_alias" in amended_source,
    "training_seed_start_guard": 'assert_eq!(seed_start, 20_000' in amended_source,
    "training_seed_count_guard": "n_seeds, 100," in amended_source
    and "amended training block is frozen at 100 seeds" in amended_source,
}
assert all(amended_source_assertions.values()), amended_source_assertions

parameter_map = []
for stratum, base in (("5bp", 0.0005), ("30bp", 0.0030)):
    for dial in DIALS:
        f0 = base * dial
        parameter_map.append(
            {
                "stratum": stratum,
                "dial_mult": dial,
                "static_fee": f0,
                "gap_f0": f0,
                "gap_alpha": 2.0,
                "gap_min_fee": f0,
                "gap_max_fee": 10.0 * f0,
                "gap_trigger_abs_bps": 0.0,
                "defensive_f0": f0,
                "defensive_alpha": 50.0,
                "defensive_min_fee": f0,
                "defensive_max_fee": 0.30,
                "defensive_trigger_abs_bps": 0.0,
            }
        )

old_artifacts = [
    LVR / "m3_training_selection.json",
    LVR / "m3_training_frontier.csv",
    LVR / "m3_validation_plan.json",
    LVR / "m3_validation_selection.json",
    LVR / "m3_final_plan.json",
    LVR / "m3_final_result.json",
]
audit = {
    "stage": "validation-grid policy-support design audit",
    "chronology_disclosure": (
        "The narrow-family validation/final blocks were already consumed before this audit request. "
        "Those artifacts are preserved for provenance but superseded; seeds 30000..30199 and "
        "90000..90399 are permanently retired and cannot validate the amended family."
    ),
    "finding": {
        "decision_triggered": True,
        "classification": "empirical gap family is narrower than the theoretical family",
        "current_dimension": 1,
        "dial_changes": ["base_fee", "min_fee", "max_fee"],
        "dial_does_not_change": ["gap_multiplier"],
        "alpha_zero_explicitly_present_in_gap_grid": False,
        "small_positive_alpha_members_present": False,
        "gap_family_contains_static_boundary": False,
        "frontier_workaround": (
            "The prior adaptive frontier used the finite union static + gap, so it was nested "
            "computationally, but the empirical gap family itself did not satisfy the theoretical inclusion."
        ),
        "interpretation": (
            "Prior results identify the fixed-alpha=2, base/cap-coupled slice only; they cannot be "
            "interpreted as a test of the two-dimensional theoretical gap family."
        ),
    },
    "current_parameterization": {
        "formula": "clip(f0 + 2*abs(stale_gap_bps)/10000, f0, 10*f0)",
        "parameter_map": parameter_map,
    },
    "physical_configuration_source_audit": source_assertions,
    "amended_runner_source_audit": amended_source_assertions,
    "amendment": {
        "status": "frozen before amended training run",
        "formula": "clip(f0 + alpha*abs(stale_gap_bps)/10000, f0, 0.30)",
        "base_fee_dial_mults": DIALS,
        "alpha_grid": ALPHAS_AMENDED,
        "fee_cap": 0.30,
        "trigger_abs_bps": 0.0,
        "properties": {
            "two_dimensional": True,
            "alpha_zero_exact_static_boundary": True,
            "low_base_small_positive_alpha_present": True,
            "cap_independent_of_base_fee": True,
        },
        "training_seed_block": {"start_inclusive": 20000, "end_exclusive": 20100, "n": 100},
        "held_out_rule": (
            "Do not use retired validation/final seeds. Freeze fresh disjoint blocks only after "
            "amended training selection and diagnostics are complete."
        ),
    },
    "superseded_artifacts_sha256": {
        str(path.relative_to(ROOT)): sha256(path) for path in old_artifacts
    },
    "source_sha256": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (TRAIN, SELECT, POLICY, AMENDED_TRAIN, Path(__file__).resolve())
    },
}
OUT.write_text(json.dumps(audit, indent=1, sort_keys=True) + "\n")
OUT.with_suffix(".sha256").write_text(f"{sha256(OUT)}  {OUT.name}\n")

lines = [
    "# validation-grid policy-support design audit",
    "",
    "## Decision",
    "",
    "The preregistered amendment trigger fires. The implemented oracle-gap grid is a one-dimensional",
    "fixed-alpha slice, not the theoretical two-dimensional `(f0, alpha)` family. Existing validation",
    "and final artifacts remain available for provenance but are superseded; their seeds are retired.",
    "",
    "## Exact current mapping",
    "",
    "`gap = clip(f0 + 2 |g_stale|, f0, 10 f0)`, with `f0 = stratum_fee * dial_mult`.",
    "The dial changes `f0`, the lower clip, and the upper clip together. It never changes alpha.",
    "There is no alpha=0 or small-positive-alpha member. The prior frontier restored nesting only by",
    "taking an external finite union with the static grid.",
    "",
    "## Frozen amendment",
    "",
    "`gap = clip(f0 + alpha |g_stale|, f0, 0.30)` with:",
    "",
    f"- base multipliers: `{DIALS}`",
    f"- alpha grid: `{ALPHAS_AMENDED}`",
    "- alpha=0 exactly reproduces every static member",
    "- the cap is independent of base fee",
    "- amended training reuses training seeds 20000..20099 only",
    "- any later held-out run must use fresh blocks, not 30000..30199 or 90000..90399",
    "",
    "## Physical configuration",
    "",
]
lines.extend(f"- {key}: `{value}`" for key, value in source_assertions.items())
REPORT.write_text("\n".join(lines) + "\n")
print(f"decision_triggered={audit['finding']['decision_triggered']}")
print(f"parameter_rows={len(parameter_map)}")
print(f"wrote {OUT} and {REPORT}")
