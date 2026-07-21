#!/usr/bin/env python3
"""Align constrained A, signed tracking difference L, and LP surplus U rankings."""

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr/workspace"
ROWS = [LVR / f"m3_amended_training_rows_shard{i}.csv.gz" for i in range(6)]
SELECTION = LVR / "m3_amended_training_selection.json"
SELECTION_HASH = LVR / "m3_amended_training_selection.sha256"
SUPPORT = LVR / "m3_joint_support.json"
FINAL = LVR / "m3_amended_final_result.json"
FINAL_HASH = LVR / "m3_amended_final_result.sha256"
PRIOR_AUDIT = LVR / "m3_constrained_divergence.csv"
PRIOR_AUDIT_HASH = LVR / "m3_constrained_divergence.csv.sha256"
OUT = LVR / "m3_loss_alignment.csv"
TRANSITIONS_OUT = LVR / "m3_loss_alignment_transition_counts.csv"
REPORT = LVR / "m3_loss_alignment_report.md"
RHOS = (0.2, 0.4, 0.6, 0.8, 0.95)
METRICS = ("a", "b", "l", "fees", "u", "s")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-8)


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
        f"{prefix}_mean_l": policy["mean_l"],
        f"{prefix}_mean_fees": policy["mean_fees"],
        f"{prefix}_mean_u": policy["mean_u"],
        f"{prefix}_mean_s": policy["mean_s"],
    }


def freeze(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing to overwrite frozen output: {path}")
    if not path.exists():
        path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n"
    )


def freeze_csv(path: Path, rows: list[dict]) -> None:
    assert rows
    fields = list(rows[0])
    assert all(list(row) == fields for row in rows)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    freeze(path, buffer.getvalue().encode())


assert SELECTION_HASH.read_text().split()[0] == sha256(SELECTION)
assert FINAL_HASH.read_text().split()[0] == sha256(FINAL)
assert PRIOR_AUDIT_HASH.read_text().split()[0] == sha256(PRIOR_AUDIT)
selection = json.loads(SELECTION.read_text())
support = json.loads(SUPPORT.read_text())
final = json.loads(FINAL.read_text())
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
    assert close(values["mean_l"], values["mean_a"] - values["mean_b"])
    assert close(values["mean_u"], values["mean_fees"] - values["mean_l"])

rows_out = []
for cell_idx in range(54):
    info = cell_info[cell_idx]
    support_cell = support_lookup[(info["stratum"], info["sigma"], info["z"])]
    s0 = means[(cell_idx, "static", 1.0, 0.0)]["mean_s"]
    policies = [
        {**specs[key], **values}
        for key, values in means.items()
        if key[0] == cell_idx and key[1] == "gap"
    ]
    assert len(policies) == 84

    for rho in RHOS:
        target = rho * s0
        feasible = [p for p in policies if p["mean_s"] >= target]
        assert feasible
        suffix = lambda p: (p["alpha"], p["dial_mult"], policy_id(p))
        pi_a = min(feasible, key=lambda p: (p["mean_a"], *suffix(p)))
        pi_l = min(feasible, key=lambda p: (p["mean_l"], *suffix(p)))
        pi_u = min(feasible, key=lambda p: (-p["mean_u"], *suffix(p)))
        pi_identity = min(
            feasible,
            key=lambda p: (-(p["mean_fees"] - p["mean_l"]), *suffix(p)),
        )
        tie_counts = {
            "argmin_a_tie_count": sum(close(p["mean_a"], pi_a["mean_a"]) for p in feasible),
            "argmin_l_tie_count": sum(close(p["mean_l"], pi_l["mean_l"]) for p in feasible),
            "argmax_u_tie_count": sum(close(p["mean_u"], pi_u["mean_u"]) for p in feasible),
            "argmax_fees_minus_l_tie_count": sum(
                close(p["mean_fees"] - p["mean_l"], pi_identity["mean_fees"] - pi_identity["mean_l"])
                for p in feasible
            ),
        }
        assert policy_id(pi_identity) == policy_id(pi_u)
        assert close(
            pi_identity["mean_fees"] - pi_identity["mean_l"],
            pi_u["mean_u"],
        )
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
                **tie_counts,
                **policy_fields("pi_a_star", pi_a),
                **policy_fields("pi_l_star", pi_l),
                **policy_fields("pi_u_star", pi_u),
                **policy_fields("pi_fees_minus_l_star", pi_identity),
                "a_vs_u_diverges": policy_id(pi_a) != policy_id(pi_u),
                "l_vs_u_diverges": policy_id(pi_l) != policy_id(pi_u),
                "a_vs_l_diverges": policy_id(pi_a) != policy_id(pi_l),
                "fees_minus_l_vs_u_diverges": policy_id(pi_identity) != policy_id(pi_u),
                "a_to_u_class_transition": f"{policy_class(pi_a)}->{policy_class(pi_u)}",
                "l_to_u_class_transition": f"{policy_class(pi_l)}->{policy_class(pi_u)}",
                "a_to_l_class_transition": f"{policy_class(pi_a)}->{policy_class(pi_l)}",
                "a_to_u_dial_transition": f"{format(pi_a['dial_mult'], 'g')}->{format(pi_u['dial_mult'], 'g')}",
                "l_to_u_dial_transition": f"{format(pi_l['dial_mult'], 'g')}->{format(pi_u['dial_mult'], 'g')}",
                "a_to_l_dial_transition": f"{format(pi_a['dial_mult'], 'g')}->{format(pi_l['dial_mult'], 'g')}",
                "pi_u_identity_residual": pi_u["mean_u"]
                - (pi_u["mean_fees"] - pi_u["mean_l"]),
            }
        )

assert len(rows_out) == 270
assert all(row["argmin_a_tie_count"] == 1 for row in rows_out)
assert all(row["argmin_l_tie_count"] == 1 for row in rows_out)
assert all(row["argmax_u_tie_count"] == 1 for row in rows_out)
assert all(row["argmax_fees_minus_l_tie_count"] == 1 for row in rows_out)
assert all(not row["fees_minus_l_vs_u_diverges"] for row in rows_out)

# Confirm the previously frozen A/U calculation exactly.
with PRIOR_AUDIT.open(newline="") as f:
    prior = {(int(r["cell_idx"]), float(r["rho"])): r for r in csv.DictReader(f)}
for row in rows_out:
    old = prior[(row["cell_idx"], row["rho"])]
    assert row["pi_a_star_id"] == old["pi_a_star_id"]
    assert row["pi_u_star_id"] == old["pi_u_star_id"]
    assert row["a_vs_u_diverges"] == (old["selection_diverges"] == "True")

comparisons = {
    "A_vs_U": ("a_vs_u_diverges", "a_to_u_class_transition", "a_to_u_dial_transition"),
    "L_vs_U": ("l_vs_u_diverges", "l_to_u_class_transition", "l_to_u_dial_transition"),
    "A_vs_L": ("a_vs_l_diverges", "a_to_l_class_transition", "a_to_l_dial_transition"),
}
transition_rows = []
for subset_name, subset in (
    ("all", rows_out),
    ("supported", [r for r in rows_out if r["support_label"] == "supported"]),
):
    for comparison, (flag, class_field, dial_field) in comparisons.items():
        divergent = [r for r in subset if r[flag]]
        for transition_type, field in (("class", class_field), ("dial", dial_field)):
            for transition, count in sorted(
                Counter(r[field] for r in divergent).items(), key=lambda x: (-x[1], x[0])
            ):
                transition_rows.append(
                    {
                        "subset": subset_name,
                        "comparison": comparison,
                        "transition_type": transition_type,
                        "transition": transition,
                        "count": count,
                        "divergent_denominator": len(divergent),
                        "share_of_divergent_cases": count / len(divergent) if divergent else 0.0,
                    }
                )

supported = [r for r in rows_out if r["support_label"] == "supported"]
assert len(supported) == 135

def count(rows: list[dict], field: str) -> int:
    return sum(bool(r[field]) for r in rows)

a_u = count(rows_out, "a_vs_u_diverges")
l_u = count(rows_out, "l_vs_u_diverges")
a_l = count(rows_out, "a_vs_l_diverges")
a_u_s = count(supported, "a_vs_u_diverges")
l_u_s = count(supported, "l_vs_u_diverges")
a_l_s = count(supported, "a_vs_l_diverges")
assert a_u == 158 and a_u_s == 79

p1 = final["policy_means"]["policy_1_lower_A"]
p2 = final["policy_means"]["policy_2"]
assert p1["b"] == 0.0 and p2["b"] == 0.0
assert close(p1["l"], p1["a"]) and close(p2["l"], p2["a"])

def pct(n: int, d: int) -> str:
    return f"{100*n/d:.1f}%"

report = f"""# validation-grid A/L/U Alignment Audit

## Scope

This final alignment audit uses only frozen validation-grid training means. It does not call
the simulator, alter a policy, or consume a seed. For each of 54 cells and five
service targets, the feasible set is the complete 84-member empirical gap
family subject to `mean_S >= rho*S0`.

The selectors are

```text
pi_A_star = argmin mean_A,
pi_L_star = argmin mean_L, where mean_L = mean_A - mean_B,
pi_U_star = argmax mean_U.
```

Ties use exactly the prior audit rule: primary objective, lower `alpha`, lower
`dial_mult`, then stable policy id. No primary-objective tie occurs for A, L,
U, or fees-minus-L in any of the 270 cases.

## Divergence counts

| comparison | all cell-targets | empirically supported |
|---|---:|---:|
| `pi_A_star != pi_U_star` | **{a_u}/270 ({pct(a_u,270)})** | **{a_u_s}/135 ({pct(a_u_s,135)})** |
| `pi_L_star != pi_U_star` | **{l_u}/270 ({pct(l_u,270)})** | **{l_u_s}/135 ({pct(l_u_s,135)})** |
| `pi_A_star != pi_L_star` | **{a_l}/270 ({pct(a_l,270)})** | **{a_l_s}/135 ({pct(a_l_s,135)})** |

The complete class and dial transitions for each disagreement are in
`m3_loss_alignment_transition_counts.csv`; every cell-target selector and
policy specification is in `m3_loss_alignment.csv`.

## Accounting benchmark

For every policy mean, the audit verifies

```text
mean_L = mean_A - mean_B,
mean_U = mean_fees - mean_L.
```

Accordingly,

```text
argmax(mean_fees - mean_L) == argmax(mean_U)
```

in **270/270** feasible cell-target states and **135/135** supported states.
There are zero accounting-benchmark selection disagreements. The current
fee-only simulator has no additional rebate or cost term; if those are added,
the implemented surplus identity must include them.

This isolates the issue: fees-minus-tracking-difference accounting reproduces
LP relative surplus exactly. Any divergence of a loss-only selector comes from
omitting fee revenue, not from failure of the accounting identity.

## Final confirmed pair

Both final policies have `B=0`, so `L=A` for each policy. The already confirmed
matched-service counterexample therefore applies to both A-only and L-only
ranking. It remains a post-hoc interpretation of the frozen final pair, not a
new held-out finding.

## Input integrity

- Training selection SHA-256: `{sha256(SELECTION)}`.
- Prior A/U audit SHA-256: `{sha256(PRIOR_AUDIT)}`.
- Final result SHA-256: `{sha256(FINAL)}`.
- Script SHA-256: `{sha256(Path(__file__).resolve())}`.
- All 518,400 training rows and frozen input hashes were checked.
"""

freeze_csv(OUT, rows_out)
freeze_csv(TRANSITIONS_OUT, transition_rows)
freeze(REPORT, report.encode())

print(f"A!=U {a_u}/270 supported {a_u_s}/135")
print(f"L!=U {l_u}/270 supported {l_u_s}/135")
print(f"A!=L {a_l}/270 supported {a_l_s}/135")
print("argmax(fees-L)==argmax(U) 270/270 supported 135/135")
print(f"wrote {OUT}, {TRANSITIONS_OUT}, {REPORT}")
