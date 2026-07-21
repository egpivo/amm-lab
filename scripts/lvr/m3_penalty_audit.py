#!/usr/bin/env python3
"""Post-hoc exact-penalty audit using only frozen M3 artifacts.

This script changes the evaluation layer only. It neither calls the simulator
nor consumes a seed. For each cell and service target it constructs the exact
finite-grid envelopes of shortfall-penalized LP surplus and gross loss.
"""

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr"
TRAINING_ROWS = [LVR / f"m3_amended_training_rows_shard{i}.csv.gz" for i in range(6)]
TRAINING_SELECTION = LVR / "m3_amended_training_selection.json"
TRAINING_SELECTION_HASH = LVR / "m3_amended_training_selection.sha256"
FINAL_RESULT = LVR / "m3_amended_final_result.json"
FINAL_RESULT_HASH = LVR / "m3_amended_final_result.sha256"
SUPPORT = LVR / "m3_joint_support.json"
BREAKPOINTS_OUT = LVR / "m3_penalty_breakpoints.csv"
PENALTIES_OUT = LVR / "m3_exact_penalties.csv"
REPORT_OUT = LVR / "m3_penalty_audit_report.md"
RHOS = (0.2, 0.4, 0.6, 0.8, 0.95)
METRICS = ("a", "b", "fees", "u", "s")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a: float, b: float, scale: float | None = None) -> bool:
    if scale is None:
        scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= 2e-10 * scale


def policy_id(policy: dict) -> str:
    dial = format(policy["dial_mult"], "g").replace(".", "p")
    alpha = format(policy["alpha"], "g").replace(".", "p")
    return f"gap_d{dial}_a{alpha}"


def policy_fields(prefix: str, policy: dict) -> dict:
    return {
        f"{prefix}_id": policy_id(policy),
        f"{prefix}_family": policy["family"],
        f"{prefix}_dial_mult": policy["dial_mult"],
        f"{prefix}_f0": policy["f0"],
        f"{prefix}_alpha": policy["alpha"],
        f"{prefix}_fee_cap": policy["fee_cap"],
        f"{prefix}_mean_a": policy["mean_a"],
        f"{prefix}_mean_b": policy["mean_b"],
        f"{prefix}_mean_fees": policy["mean_fees"],
        f"{prefix}_mean_u": policy["mean_u"],
        f"{prefix}_mean_s": policy["mean_s"],
    }


def freeze_csv(path: Path, rows: list[dict]) -> None:
    assert rows
    fieldnames = list(rows[0])
    assert all(list(row) == fieldnames for row in rows)
    lines = []
    from io import StringIO

    buf = StringIO(newline="")
    out = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    out.writeheader()
    out.writerows(rows)
    payload = buf.getvalue().encode()
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing to overwrite frozen output: {path}")
    if not path.exists():
        path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n"
    )


def freeze_text(path: Path, text: str) -> None:
    payload = text.encode()
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing to overwrite frozen output: {path}")
    if not path.exists():
        path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n"
    )


def upper_envelope(lines: list[dict], objective: str) -> list[dict]:
    """Return exact analytic envelope segments on lambda >= 0.

    Surplus lines maximize U - lambda*shortfall. Loss lines minimize
    A + lambda*shortfall, implemented by maximizing its negative.
    """

    assert objective in ("surplus", "loss")
    transformed = []
    for line in lines:
        if objective == "surplus":
            intercept = line["mean_u"]
            slope = -line["shortfall"]
        else:
            intercept = -line["mean_a"]
            slope = -line["shortfall"]
        transformed.append({**line, "hull_intercept": intercept, "hull_slope": slope})

    # Equal-slope lines: only the largest transformed intercept can appear.
    by_slope: dict[float, list[dict]] = defaultdict(list)
    for line in transformed:
        by_slope[line["hull_slope"]].append(line)
    candidates = []
    for slope in sorted(by_slope):
        group = by_slope[slope]
        best_intercept = max(x["hull_intercept"] for x in group)
        best = sorted(
            (x for x in group if close(x["hull_intercept"], best_intercept)),
            key=lambda x: x["policy_id"],
        )[0]
        candidates.append(best)

    hull: list[dict] = []
    starts: list[float] = []
    for line in candidates:
        start = -math.inf
        while hull:
            prev = hull[-1]
            denominator = line["hull_slope"] - prev["hull_slope"]
            assert denominator > 0
            start = (prev["hull_intercept"] - line["hull_intercept"]) / denominator
            if starts[-1] == -math.inf or (
                start > starts[-1] and not close(start, starts[-1])
            ):
                break
            hull.pop()
            starts.pop()
        if not hull:
            start = -math.inf
        hull.append(line)
        starts.append(start)

    segments = []
    for i, line in enumerate(hull):
        raw_start = starts[i]
        raw_end = starts[i + 1] if i + 1 < len(starts) else math.inf
        if raw_end < 0 and not close(raw_end, 0.0):
            continue
        start = max(0.0, raw_start)
        end = raw_end
        if math.isfinite(end) and (end < start or close(end, start)):
            continue
        if objective == "surplus":
            value_at_start = line["mean_u"] - start * line["shortfall"]
            all_values = [x["mean_u"] - start * x["shortfall"] for x in lines]
            optimum = max(all_values)
        else:
            value_at_start = line["mean_a"] + start * line["shortfall"]
            all_values = [x["mean_a"] + start * x["shortfall"] for x in lines]
            optimum = min(all_values)
        scale = max(1.0, abs(optimum), *(abs(x) for x in all_values))
        assert close(value_at_start, optimum, scale)
        ties = sorted(
            x["policy_id"] for x, value in zip(lines, all_values) if close(value, optimum, scale)
        )
        cooptimal = sorted(
            x["policy_id"]
            for x in lines
            if close(x["shortfall"], line["shortfall"])
            and close(
                x["mean_u"] if objective == "surplus" else x["mean_a"],
                line["mean_u"] if objective == "surplus" else line["mean_a"],
            )
        )
        segments.append(
            {
                "lambda_start": start,
                "lambda_end": None if math.isinf(end) else end,
                "policy": line,
                "ties_at_start": ties,
                "cooptimal": cooptimal,
            }
        )

    assert segments and segments[0]["lambda_start"] == 0.0
    assert segments[-1]["lambda_end"] is None
    for left, right in zip(segments, segments[1:]):
        assert close(left["lambda_end"], right["lambda_start"])
    return segments


def segment_after(segments: list[dict], threshold: float) -> dict:
    scale = max(1.0, abs(threshold))
    eps = 2e-10 * scale
    for segment in segments:
        end = segment["lambda_end"]
        if segment["lambda_start"] <= threshold + eps and (end is None or end > threshold + eps):
            return segment
    # If threshold is itself a transition within floating precision, use the
    # interval beginning at that exact analytic breakpoint.
    for segment in segments:
        if segment["lambda_start"] >= threshold - eps:
            return segment
    raise AssertionError((threshold, segments))


def self_test() -> None:
    lines = [
        {"policy_id": "high", "mean_u": 10.0, "mean_a": 1.0, "shortfall": 5.0},
        {"policy_id": "feasible", "mean_u": 4.0, "mean_a": 3.0, "shortfall": 0.0},
    ]
    surplus = upper_envelope(lines, "surplus")
    assert len(surplus) == 2
    assert close(surplus[0]["lambda_end"], 1.2)
    assert surplus[1]["policy"]["policy_id"] == "feasible"
    loss = upper_envelope(lines, "loss")
    assert len(loss) == 2
    assert close(loss[0]["lambda_end"], 0.4)
    assert loss[1]["policy"]["policy_id"] == "feasible"


self_test()

assert TRAINING_SELECTION_HASH.read_text().split()[0] == sha256(TRAINING_SELECTION)
assert FINAL_RESULT_HASH.read_text().split()[0] == sha256(FINAL_RESULT)
selection = json.loads(TRAINING_SELECTION.read_text())
support = json.loads(SUPPORT.read_text())
final = json.loads(FINAL_RESULT.read_text())
assert selection["seed_block"] == {"start_inclusive": 20000, "end_exclusive": 20100, "n": 100}
assert final["integrity"]["seed_start_inclusive"] == 91000
assert final["integrity"]["seed_end_exclusive"] == 91400

for path in TRAINING_ROWS:
    relative = str(path.relative_to(ROOT))
    assert selection["input_sha256"][relative] == sha256(path)

support_lookup = {
    (x["stratum"], x["sigma"], x["z"]): x for x in support["market_state_cells"]
}

sums = defaultdict(lambda: {metric: 0.0 for metric in METRICS})
counts = Counter()
specs = {}
cell_info = {}
row_count = 0
for path in TRAINING_ROWS:
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            row_count += 1
            cell_idx = int(row["cell_idx"])
            family = row["family"]
            dial = float(row["dial_mult"])
            alpha = float(row["alpha"])
            key = (cell_idx, family, dial, alpha)
            counts[key] += 1
            for metric in METRICS:
                sums[key][metric] += float(row[metric])
            specs[key] = {
                "family": family,
                "dial_mult": dial,
                "f0": float(row["f0"]),
                "alpha": alpha,
                "fee_cap": float(row["fee_cap"]),
            }
            cell_info[cell_idx] = {
                "cell_idx": cell_idx,
                "stratum": row["stratum"],
                "sigma": float(row["sigma"]),
                "z": float(row["z"]),
                "speed": row["speed"],
            }

assert row_count == 518_400
assert len(counts) == 54 * 96
assert set(counts.values()) == {100}
means = {
    key: {f"mean_{metric}": total / counts[key] for metric, total in values.items()}
    for key, values in sums.items()
}
for key, values in means.items():
    assert close(
        values["mean_u"],
        values["mean_fees"] - values["mean_a"] + values["mean_b"],
    )

for cell_idx in range(54):
    for dial in selection["grid"]["dial_mults"]:
        static = means[(cell_idx, "static", dial, 0.0)]
        alias = means[(cell_idx, "gap", dial, 0.0)]
        assert static == alias

breakpoint_rows = []
penalty_rows = []
lambda_stars = []
surplus_segments_per_case = []
loss_segments_per_case = []
loss_diff_above_count = 0

for cell_idx in range(54):
    info = cell_info[cell_idx]
    support_cell = support_lookup[(info["stratum"], info["sigma"], info["z"])]
    s0 = means[(cell_idx, "static", 1.0, 0.0)]["mean_s"]
    policies = []
    for (ci, family, dial, alpha), policy_means in means.items():
        if ci != cell_idx or family != "gap":
            continue
        policy = {**specs[(ci, family, dial, alpha)], **policy_means}
        policy["policy_id"] = policy_id(policy)
        policies.append(policy)
    policies.sort(key=lambda x: (x["alpha"], x["dial_mult"]))
    assert len(policies) == 84

    for rho in RHOS:
        target = rho * s0
        lines = [{**p, "shortfall": max(target - p["mean_s"], 0.0)} for p in policies]
        feasible = [p for p in lines if p["shortfall"] == 0.0]
        assert feasible
        optimum = max(feasible, key=lambda p: (p["mean_u"], -p["alpha"], -p["dial_mult"]))
        u_star = optimum["mean_u"]
        ratios = []
        for p in lines:
            if p["shortfall"] > 0:
                ratios.append((max(p["mean_u"] - u_star, 0.0) / p["shortfall"], p))
        lambda_star = max((ratio for ratio, _ in ratios), default=0.0)
        binding = sorted(
            p["policy_id"] for ratio, p in ratios if close(ratio, lambda_star)
        )
        lambda_stars.append(lambda_star)

        surplus_segments = upper_envelope(lines, "surplus")
        loss_segments = upper_envelope(lines, "loss")
        surplus_segments_per_case.append(len(surplus_segments))
        loss_segments_per_case.append(len(loss_segments))
        surplus_after = segment_after(surplus_segments, lambda_star)["policy"]
        loss_after = segment_after(loss_segments, lambda_star)["policy"]
        surplus_recovers = (
            surplus_after["shortfall"] == 0.0 and close(surplus_after["mean_u"], u_star)
        )
        assert surplus_recovers
        loss_diff_above = loss_after["policy_id"] != optimum["policy_id"]
        loss_diff_above_count += loss_diff_above

        common = {
            "cell_idx": cell_idx,
            "stratum": info["stratum"],
            "sigma": info["sigma"],
            "z": info["z"],
            "speed": info["speed"],
            "support_label": support_cell["support_label"],
            "observed_pool_weeks": support_cell["observed_pool_weeks"],
            "rho": rho,
            "training_S0": s0,
            "service_target": target,
        }
        for objective, segments in (("penalized_surplus_max", surplus_segments), ("penalized_loss_min", loss_segments)):
            for segment_index, segment in enumerate(segments):
                p = segment["policy"]
                breakpoint_rows.append(
                    {
                        **common,
                        "objective": objective,
                        "segment_index": segment_index,
                        "lambda_start_inclusive": segment["lambda_start"],
                        "lambda_end_exclusive": (
                            "inf" if segment["lambda_end"] is None else segment["lambda_end"]
                        ),
                        "ties_at_lambda_start": ";".join(segment["ties_at_start"]),
                        "cooptimal_throughout_interval": ";".join(segment["cooptimal"]),
                        **policy_fields("selected", p),
                        "service_shortfall": p["shortfall"],
                        "line_intercept": p["mean_u"] if objective == "penalized_surplus_max" else p["mean_a"],
                        "line_slope": -p["shortfall"] if objective == "penalized_surplus_max" else p["shortfall"],
                        "feasible": p["shortfall"] == 0.0,
                        "is_constrained_surplus_optimum": (
                            p["shortfall"] == 0.0 and close(p["mean_u"], u_star)
                        ),
                    }
                )

        penalty_rows.append(
            {
                **common,
                "n_gap_policies": len(lines),
                "n_feasible_policies": len(feasible),
                "lambda_star": lambda_star,
                "lambda_star_binding_infeasible_policies": ";".join(binding),
                **policy_fields("constrained_optimum", optimum),
                "surplus_envelope_segments": len(surplus_segments),
                "surplus_policy_lambda0": surplus_segments[0]["policy"]["policy_id"],
                "surplus_policy_strictly_above_lambda_star": surplus_after["policy_id"],
                "penalized_surplus_recovers_constrained_optimum": surplus_recovers,
                "loss_envelope_segments": len(loss_segments),
                "loss_policy_lambda0": loss_segments[0]["policy"]["policy_id"],
                "loss_policy_strictly_above_lambda_star": loss_after["policy_id"],
                "penalized_loss_differs_above_lambda_star": loss_diff_above,
                "penalized_loss_ever_differs_from_constrained_optimum": any(
                    segment["policy"]["policy_id"] != optimum["policy_id"]
                    for segment in loss_segments
                ),
            }
        )

assert len(penalty_rows) == 270
assert len(breakpoint_rows) == sum(surplus_segments_per_case) + sum(loss_segments_per_case)

p1 = final["policy_means"]["policy_1_lower_A"]
p2 = final["policy_means"]["policy_2"]
delta = {metric: p1[metric] - p2[metric] for metric in ("a", "b", "fees", "u", "s")}
identity_rhs = delta["fees"] - delta["a"] + delta["b"]
assert close(delta["u"], identity_rhs)
assert p1["a"] < p2["a"] and p1["u"] < p2["u"]
service_zero_penalty_max = min(p1["s"], p2["s"])

lambda_positive = sum(x > 0 for x in lambda_stars)
lambda_zero = len(lambda_stars) - lambda_positive
support_counts = Counter(row["support_label"] for row in penalty_rows)
surplus_segment_count = sum(surplus_segments_per_case)
loss_segment_count = sum(loss_segments_per_case)

report = f"""# M3 Exact-Penalty Audit

## Status and scope

This is a post-hoc, theory-guided re-analysis of the frozen M3 evaluation
layer. It is not part of the original preregistration. It uses only the frozen
training rows (`20000..20099`), validation selection, and final result
(`91000..91399`). It does not call the simulator, consume a seed, alter the
calibrated hazards, change the policy grid, or modify a held-out artifact.

Loss-only ranking is treated as a diagnostic. The normative primary object is
the service-constrained LP-surplus frontier

```text
F(s) = max_pi mean_U(pi) subject to mean_S(pi) >= s.
```

The operational finite-grid representation is

```text
J_U(lambda,s;pi) = mean_U(pi) - lambda*max(s-mean_S(pi),0).
```

The separately audited, insufficient loss formulation is

```text
J_A(lambda,s;pi) = mean_A(pi) + lambda*max(s-mean_S(pi),0).
```

## Exact finite-grid penalty audit

The audit covers 54 cells, five service targets per cell, and all 84 members of
the frozen empirical gap family, including the 12 `alpha=0` static-boundary
members. It verifies 518,400 training rows and the identity
`mean_U = mean_fees - mean_A + mean_B` for every policy.

- Cell-target cases: 270 ({support_counts['supported']} supported,
  {support_counts['sparse']} sparse, {support_counts['unobserved']} unobserved).
- Positive `lambda_star`: {lambda_positive}; zero `lambda_star`: {lambda_zero}.
- Penalized-surplus recovery for every `lambda > lambda_star`: 270/270.
- Penalized-loss policy differs from the constrained surplus optimum just
  above `lambda_star`: {loss_diff_above_count}/270.
- Analytic surplus-envelope intervals: {surplus_segment_count}.
- Analytic loss-envelope intervals: {loss_segment_count}.

Every breakpoint is an analytic intersection of two finite-policy lines. No
arbitrary lambda grid is used. The complete interval tables, including ties at
each left endpoint, are in `m3_penalty_breakpoints.csv`. The constrained
optimum and exact `lambda_star` for every cell-target case are in
`m3_exact_penalties.csv`.

The result is an exact finite-policy-grid statement. It is not a claim of
strong duality for a continuous or convex policy space.

## Final matched-service counterexample

The frozen final orientation is policy 1 (gap: `f0=0.001`, `alpha=2`) minus
policy 2 (static: `f0=0.0035`, `alpha=0`).

| metric | policy 1 | policy 2 | delta (1-2) |
|---|---:|---:|---:|
| A | {p1['a']:.6f} | {p2['a']:.6f} | {delta['a']:.6f} |
| B | {p1['b']:.6f} | {p2['b']:.6f} | {delta['b']:.6f} |
| fees | {p1['fees']:.6f} | {p2['fees']:.6f} | {delta['fees']:.6f} |
| U | {p1['u']:.6f} | {p2['u']:.6f} | {delta['u']:.6f} |
| S | {p1['s']:.6f} | {p2['s']:.6f} | {delta['s']:.6f} |

Numerically,

```text
delta_U = delta_fees - delta_A + delta_B
        = {delta['fees']:.6f} - ({delta['a']:.6f}) + {delta['b']:.6f}
        = {identity_rhs:.6f}.
```

For every service requirement

```text
s <= min(S1,S2) = {service_zero_penalty_max:.6f},
```

both shortfall penalties are exactly zero. Therefore, for every
`lambda >= 0`,

```text
J_A(lambda,s;policy 1) = A1 < A2 = J_A(lambda,s;policy 2),
```

while `U1 < U2`, so LP relative surplus prefers policy 2. A service-shortfall
penalty can remove an inactivity incentive when it binds, but it cannot repair
a loss ranking that omits fee revenue or favorable execution components once
both policies satisfy the service requirement. This is not a failure of the
fees-minus-tracking-difference accounting identity.

The selected final cell is empirically unobserved and remains a
mechanism-boundary counterexample, not an empirically supported deployment
claim.

## Artifact integrity

- Training selection SHA-256: `{sha256(TRAINING_SELECTION)}`.
- Final result SHA-256: `{sha256(FINAL_RESULT)}`.
- Audit script SHA-256: `{sha256(Path(__file__).resolve())}`.
- Row-level seed blocks and all frozen input hashes were checked before output.
"""

freeze_csv(PENALTIES_OUT, penalty_rows)
freeze_csv(BREAKPOINTS_OUT, breakpoint_rows)
freeze_text(REPORT_OUT, report)

print(f"cases={len(penalty_rows)} breakpoints={len(breakpoint_rows)}")
print(f"lambda_star_positive={lambda_positive} zero={lambda_zero}")
print(f"surplus_recovery={sum(r['penalized_surplus_recovers_constrained_optimum'] for r in penalty_rows)}/270")
print(f"loss_differs_above_lambda_star={loss_diff_above_count}/270")
print(f"final_zero_penalty_service_max={service_zero_penalty_max:.6f}")
print(f"final_identity_residual={delta['u'] - identity_rhs:.12g}")
print(f"wrote {PENALTIES_OUT}, {BREAKPOINTS_OUT}, {REPORT_OUT}")
