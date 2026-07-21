#!/usr/bin/env python3
"""Constrained loss-vs-surplus selection divergence on frozen M3 training means."""

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr"
ROWS = [LVR / f"m3_amended_training_rows_shard{i}.csv.gz" for i in range(6)]
SELECTION = LVR / "m3_amended_training_selection.json"
SELECTION_HASH = LVR / "m3_amended_training_selection.sha256"
SUPPORT = LVR / "m3_joint_support.json"
PENALTIES = LVR / "m3_exact_penalties.csv"
PENALTIES_HASH = LVR / "m3_exact_penalties.csv.sha256"
OUT = LVR / "m3_constrained_divergence.csv"
TRANSITIONS_OUT = LVR / "m3_constrained_transition_counts.csv"
NORMALIZED_OUT = LVR / "m3_normalized_penalties.csv"
REPORT = LVR / "m3_constrained_divergence_report.md"
RHOS = (0.2, 0.4, 0.6, 0.8, 0.95)
METRICS = ("a", "b", "fees", "u", "s")
V0 = 40_000_000.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-9)


def policy_id(policy: dict) -> str:
    dial = format(policy["dial_mult"], "g").replace(".", "p")
    alpha = format(policy["alpha"], "g").replace(".", "p")
    return f"gap_d{dial}_a{alpha}"


def policy_class(policy: dict) -> str:
    return "static_boundary" if policy["alpha"] == 0.0 else "adaptive_gap"


def policy_fields(prefix: str, policy: dict) -> dict:
    return {
        f"{prefix}_id": policy_id(policy),
        f"{prefix}_class": policy_class(policy),
        f"{prefix}_dial_mult": policy["dial_mult"],
        f"{prefix}_f0": policy["f0"],
        f"{prefix}_alpha": policy["alpha"],
        f"{prefix}_mean_a": policy["mean_a"],
        f"{prefix}_mean_b": policy["mean_b"],
        f"{prefix}_mean_fees": policy["mean_fees"],
        f"{prefix}_mean_u": policy["mean_u"],
        f"{prefix}_mean_s": policy["mean_s"],
    }


def freeze_csv(path: Path, rows: list[dict]) -> None:
    assert rows
    fields = list(rows[0])
    assert all(list(row) == fields for row in rows)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    freeze(path, buffer.getvalue().encode())


def freeze(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing to overwrite frozen output: {path}")
    if not path.exists():
        path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n"
    )


def quantile(values: list[float], p: float) -> float:
    xs = sorted(values)
    pos = p * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": quantile(values, 0.10),
        "p90": quantile(values, 0.90),
    }


assert SELECTION_HASH.read_text().split()[0] == sha256(SELECTION)
assert PENALTIES_HASH.read_text().split()[0] == sha256(PENALTIES)
selection = json.loads(SELECTION.read_text())
support = json.loads(SUPPORT.read_text())
assert selection["seed_block"] == {
    "start_inclusive": 20000,
    "end_exclusive": 20100,
    "n": 100,
}
for path in ROWS:
    assert selection["input_sha256"][str(path.relative_to(ROOT))] == sha256(path)

support_lookup = {
    (x["stratum"], x["sigma"], x["z"]): x for x in support["market_state_cells"]
}
sums = defaultdict(lambda: {metric: 0.0 for metric in METRICS})
counts = Counter()
specs = {}
cell_info = {}
row_count = 0
for path in ROWS:
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            row_count += 1
            key = (
                int(row["cell_idx"]),
                row["family"],
                float(row["dial_mult"]),
                float(row["alpha"]),
            )
            counts[key] += 1
            for metric in METRICS:
                sums[key][metric] += float(row[metric])
            specs[key] = {
                "family": row["family"],
                "dial_mult": float(row["dial_mult"]),
                "f0": float(row["f0"]),
                "alpha": float(row["alpha"]),
                "fee_cap": float(row["fee_cap"]),
            }
            cell_info[key[0]] = {
                "cell_idx": key[0],
                "stratum": row["stratum"],
                "sigma": float(row["sigma"]),
                "z": float(row["z"]),
                "speed": row["speed"],
            }

assert row_count == 518_400
assert len(counts) == 54 * 96 and set(counts.values()) == {100}
means = {
    key: {f"mean_{metric}": value / counts[key] for metric, value in totals.items()}
    for key, totals in sums.items()
}
for values in means.values():
    assert close(
        values["mean_u"],
        values["mean_fees"] - values["mean_a"] + values["mean_b"],
    )

rows_out = []
for cell_idx in range(54):
    info = cell_info[cell_idx]
    support_cell = support_lookup[(info["stratum"], info["sigma"], info["z"])]
    s0 = means[(cell_idx, "static", 1.0, 0.0)]["mean_s"]
    policies = []
    for key, values in means.items():
        ci, family, _, _ = key
        if ci == cell_idx and family == "gap":
            policies.append({**specs[key], **values})
    assert len(policies) == 84

    for rho in RHOS:
        target = rho * s0
        feasible = [p for p in policies if p["mean_s"] >= target]
        assert feasible
        # Deterministic ties: primary objective, then lower alpha, lower dial,
        # then stable policy id. No exact primary-objective ties occur here.
        pi_a = min(
            feasible,
            key=lambda p: (p["mean_a"], p["alpha"], p["dial_mult"], policy_id(p)),
        )
        pi_u = min(
            feasible,
            key=lambda p: (-p["mean_u"], p["alpha"], p["dial_mult"], policy_id(p)),
        )
        a_ties = [p for p in feasible if close(p["mean_a"], pi_a["mean_a"])]
        u_ties = [p for p in feasible if close(p["mean_u"], pi_u["mean_u"])]
        delta = {
            metric: pi_a[f"mean_{metric}"] - pi_u[f"mean_{metric}"]
            for metric in METRICS
        }
        assert delta["a"] <= 1e-8 and delta["u"] <= 1e-8
        assert close(delta["u"], delta["fees"] - delta["a"] + delta["b"])
        rows_out.append(
            {
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
                "n_feasible_policies": len(feasible),
                "tie_break_rule": "primary objective; lower alpha; lower dial_mult; policy_id",
                "argmin_a_tie_count": len(a_ties),
                "argmax_u_tie_count": len(u_ties),
                **policy_fields("pi_a_star", pi_a),
                **policy_fields("pi_u_star", pi_u),
                "selection_diverges": policy_id(pi_a) != policy_id(pi_u),
                "class_transition": f"{policy_class(pi_a)}->{policy_class(pi_u)}",
                "dial_transition": f"{format(pi_a['dial_mult'], 'g')}->{format(pi_u['dial_mult'], 'g')}",
                "delta_a_pi_a_minus_pi_u": delta["a"],
                "delta_b_pi_a_minus_pi_u": delta["b"],
                "delta_fees_pi_a_minus_pi_u": delta["fees"],
                "delta_u_pi_a_minus_pi_u": delta["u"],
                "delta_s_pi_a_minus_pi_u": delta["s"],
                "accounting_identity_residual": delta["u"]
                - (delta["fees"] - delta["a"] + delta["b"]),
            }
        )

assert len(rows_out) == 270
assert all(row["argmin_a_tie_count"] == 1 for row in rows_out)
assert all(row["argmax_u_tie_count"] == 1 for row in rows_out)

transition_rows = []
for subset_name, subset in (
    ("all", rows_out),
    ("supported", [r for r in rows_out if r["support_label"] == "supported"]),
):
    for transition_type, field in (("class", "class_transition"), ("dial", "dial_transition")):
        counter = Counter(r[field] for r in subset)
        for transition, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            transition_rows.append(
                {
                    "subset": subset_name,
                    "transition_type": transition_type,
                    "transition": transition,
                    "count": count,
                    "denominator": len(subset),
                    "share": count / len(subset),
                }
            )

normalized_rows = []
with PENALTIES.open(newline="") as f:
    for row in csv.DictReader(f):
        s0 = float(row["training_S0"])
        raw = float(row["lambda_star"])
        normalized_rows.append(
            {
                "cell_idx": int(row["cell_idx"]),
                "stratum": row["stratum"],
                "sigma": float(row["sigma"]),
                "z": float(row["z"]),
                "speed": row["speed"],
                "support_label": row["support_label"],
                "rho": float(row["rho"]),
                "V0": V0,
                "training_S0": s0,
                "lambda_star_raw_U_per_S": raw,
                "lambda_tilde_star": raw * s0 / V0,
                "normalization": "U_tilde=U/V0; S_tilde=S/S0",
                "interpretation": "finite-grid training threshold; not an identified shadow price",
            }
        )
assert len(normalized_rows) == 270

all_div = [r for r in rows_out if r["selection_diverges"]]
supported_rows = [r for r in rows_out if r["support_label"] == "supported"]
supported_div = [r for r in supported_rows if r["selection_diverges"]]
assert len(supported_rows) == 135

def summary_table(rows: list[dict]) -> str:
    labels = (
        ("delta A", "delta_a_pi_a_minus_pi_u"),
        ("delta fees", "delta_fees_pi_a_minus_pi_u"),
        ("delta U", "delta_u_pi_a_minus_pi_u"),
        ("delta S", "delta_s_pi_a_minus_pi_u"),
    )
    lines = ["| difference (pi_A - pi_U) | mean | median | p10 | p90 |", "|---|---:|---:|---:|---:|"]
    for label, field in labels:
        stats = summarize([r[field] for r in rows])
        lines.append(
            f"| {label} | {stats['mean']:.3f} | {stats['median']:.3f} | "
            f"{stats['p10']:.3f} | {stats['p90']:.3f} |"
        )
    return "\n".join(lines)

class_all = Counter(r["class_transition"] for r in rows_out)
class_supported = Counter(r["class_transition"] for r in supported_rows)
top_dials = Counter(r["dial_transition"] for r in rows_out).most_common(12)
top_dials_supported = Counter(r["dial_transition"] for r in supported_rows).most_common(12)

report = f"""# M3 Constrained-Ranking Divergence

## Scope

This is a post-hoc evaluation-layer calculation on frozen M3 training means.
No simulator was called and no new seed was consumed. In each cell and for each
`rho`, the feasible set is the complete 84-member empirical gap family subject
to `mean_S >= rho*S0`, including its `alpha=0` static boundary.

The two selectors are

```text
pi_A_star = argmin mean_A over the feasible set,
pi_U_star = argmax mean_U over the feasible set.
```

Ties are resolved deterministically by the primary objective, lower `alpha`,
lower `dial_mult`, and stable policy id. There are no exact primary-objective
ties in the 270 audited cases.

## Main result

- Constrained selection divergence: **{len(all_div)}/270**.
- Empirically supported subset: **{len(supported_div)}/135**.
- Secondary local-lambda diagnostic from the earlier exact-penalty audit:
  penalized loss differs from the constrained surplus optimum just above the
  surplus `lambda_star` in 242/270 cases. That number mixes infeasible-policy
  exclusion with within-feasible-set ranking and is not the primary statistic.

Class transitions, all 270 cases:

{chr(10).join(f'- `{k}`: {v}/270' for k, v in sorted(class_all.items()))}

Class transitions, supported 135 cases:

{chr(10).join(f'- `{k}`: {v}/135' for k, v in sorted(class_supported.items()))}

The complete class and dial transition counts are in
`m3_constrained_transition_counts.csv`. The most common all-cell dial
transitions are {', '.join(f'`{k}` ({v})' for k, v in top_dials)}. In the
supported subset they are
{', '.join(f'`{k}` ({v})' for k, v in top_dials_supported)}.

## Selected-policy differences

All 270 cell-target cases:

{summary_table(rows_out)}

Empirically supported 135 cases:

{summary_table(supported_rows)}

Each cell-target row, including both selected policy specifications and the
differences in `A`, `B`, fees, `U`, and `S`, is in
`m3_constrained_divergence.csv`. The identity
`delta U = delta fees - delta A + delta B` is checked row by row.

## Penalty normalization and boundaries

The raw `lambda_star` has units of surplus per unit service and must not be
compared across cells without normalization. With initial pool value
`V0={V0:.0f}`, define

```text
U_tilde = U/V0,
S_tilde = S/S0,
lambda_tilde_star = lambda_star*S0/V0.
```

All 270 normalized thresholds are in `m3_normalized_penalties.csv`.
`lambda_star` is a finite-grid, training-block exact-penalty threshold. It
changes with the policy grid, service definition, and estimated means; it is
not an identified preference parameter, market price of service, or structural
shadow price.

The constraint and penalty use expected service,
`(s-E[S])_+`, not expected pathwise shortfall `E[(s-S)_+]`. They enforce a mean
service requirement, not episode-level reliability or a tail-service
guarantee.

## Input integrity

- Training selection SHA-256: `{sha256(SELECTION)}`.
- Exact-penalty table SHA-256: `{sha256(PENALTIES)}`.
- Script SHA-256: `{sha256(Path(__file__).resolve())}`.
- Frozen training row hashes and all 518,400 row identities were verified.
"""

freeze_csv(OUT, rows_out)
freeze_csv(TRANSITIONS_OUT, transition_rows)
freeze_csv(NORMALIZED_OUT, normalized_rows)
freeze(REPORT, report.encode())

print(f"divergence={len(all_div)}/270 supported={len(supported_div)}/135")
print(f"class_transitions={dict(sorted(class_all.items()))}")
print(f"supported_class_transitions={dict(sorted(class_supported.items()))}")
print(f"wrote {OUT}, {TRANSITIONS_OUT}, {NORMALIZED_OUT}, {REPORT}")
