#!/usr/bin/env python3
"""Aggregate paired-ledger decomposition for the validation-grid final held-out block."""

import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t as student_t


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr/workspace"
PLAN_PATH = LVR / "m3_amended_final_plan.json"
PLAN_HASH_PATH = LVR / "m3_amended_final_plan.sha256"
SEEDS_PATH = LVR / "m3_amended_final_decomposition_seeds.csv.gz"
AUDIT_PATH = LVR / "m3_amended_final_decomposition_audit.json"
RESULT_PATH = LVR / "m3_amended_final_decomposition_result.json"
BATCH_SIZE = 25

COMPONENTS = (
    ("delta_qty_c_total", "common_fill_quantity", "total"),
    ("delta_sev_c_total", "common_fill_severity", "total"),
    ("delta_entry_total", "entry", "total"),
    ("delta_exit_total", "exit", "total"),
    ("delta_a_total", "total", "total"),
    ("delta_qty_c_fund", "common_fill_quantity", "fundamental"),
    ("delta_sev_c_fund", "common_fill_severity", "fundamental"),
    ("delta_entry_fund", "entry", "fundamental"),
    ("delta_exit_fund", "exit", "fundamental"),
    ("delta_a_fund", "total", "fundamental"),
    ("delta_qty_c_arb", "common_fill_quantity", "arbitrage"),
    ("delta_sev_c_arb", "common_fill_severity", "arbitrage"),
    ("delta_entry_arb", "entry", "arbitrage"),
    ("delta_exit_arb", "exit", "arbitrage"),
    ("delta_a_arb", "total", "arbitrage"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def paired_stats(xs: list[float]) -> dict:
    n = len(xs)
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    se = sd / math.sqrt(n)
    one_sided_half = float(student_t.ppf(0.95, n - 1)) * se
    two_sided_half = float(student_t.ppf(0.975, n - 1)) * se
    return {
        "n": n,
        "mean": m,
        "sd": sd,
        "se": se,
        "one_sided_95_lower": m - one_sided_half,
        "one_sided_95_upper": m + one_sided_half,
        "two_sided_95_halfwidth": two_sided_half,
    }


def share_pct(component: float, total: float) -> float | None:
    if total == 0 or not math.isfinite(total):
        return None
    if component * total < 0:
        return None
    return 100.0 * component / total


plan_hash = sha256(PLAN_PATH)
assert PLAN_HASH_PATH.read_text().split()[0] == plan_hash
plan = json.loads(PLAN_PATH.read_text())
seed_block = plan["final_seed_block"]
seeds = range(seed_block["start_inclusive"], seed_block["end_exclusive"])
assert len(seeds) == 400

rows = []
with gzip.open(SEEDS_PATH, "rt", newline="") as f:
    for raw in csv.DictReader(f):
        row = {k: float(raw[k]) if k.startswith("delta_") or k.endswith("_err") else raw[k]
               for k in raw}
        row["seed"] = int(float(raw["seed"]))
        row["reconstruct_ok"] = raw["reconstruct_ok"] in {"true", "True", "1"}
        for k in raw:
            if k.startswith("n_"):
                row[k] = int(float(raw[k]))
        rows.append(row)

assert len(rows) == 400
assert all(r["reconstruct_ok"] for r in rows)

# Per-seed exact reconstruction (policy1 gap minus policy2 static).
for r in rows:
    parts = (
        r["delta_qty_c_total"],
        r["delta_sev_c_total"],
        r["delta_entry_total"],
        r["delta_exit_total"],
    )
    err = abs(r["delta_a_total"] - sum(parts))
    tol = 1e-6 * max(1.0, abs(r["delta_a_total"]))
    assert err <= tol, (r["seed"], err, parts, r["delta_a_total"])

stats = {}
for col, label, leg in COMPONENTS:
    xs = [r[col] for r in rows]
    entry = paired_stats(xs)
    entry["component"] = label
    entry["leg"] = leg
    stats[f"{leg}:{label}"] = entry

total_mean = stats["total:total"]["mean"]
for key in list(stats):
    if key.endswith(":total") and key != "total:total":
        leg = key.split(":")[0]
        comp = stats[key]["mean"]
        stats[key]["share_of_delta_a_pct"] = share_pct(comp, total_mean)

table_rows = []
for comp_key, comp_label in (
    ("common_fill_quantity", "Common-fill quantity"),
    ("common_fill_severity", "Common-fill severity"),
    ("entry", "Entry"),
    ("exit", "Exit"),
    ("total", "Total"),
):
    s = stats[f"total:{comp_key}"]
    table_rows.append(
        {
            "component": comp_label,
            "estimate": round(s["mean"]),
            "share_of_delta_a_pct": (
                round(s["share_of_delta_a_pct"], 1)
                if "share_of_delta_a_pct" in s and s["share_of_delta_a_pct"] is not None
                else (100.0 if comp_key == "total" else None)
            ),
            "two_sided_95_halfwidth": round(s["two_sided_95_halfwidth"]),
        }
    )

result = {
    "step": "validation-grid final paired-ledger decomposition (gap minus static)",
    "final_plan_sha256": plan_hash,
    "seeds_sha256": sha256(SEEDS_PATH),
    "candidate_id": plan["candidate"]["candidate_id"],
    "policy_1": "gap f0=0.001 alpha=2",
    "policy_0": "static f0=0.0035",
    "convention": "delta = policy_1 - policy_2; entry = gap-only fills; exit = static-only fills",
    "deterministic_engine_note": "delta_inc_C is identically zero when both policies fill on C",
    "ledger_audit_path": str(AUDIT_PATH.relative_to(ROOT)),
    "per_seed_reconstruction": {
        "all_ok": True,
        "max_err": max(r["max_reconstruct_err"] for r in rows),
    },
    "support_counts_mean": {
        "n_common_fund": statistics.fmean(r["n_common_fund"] for r in rows),
        "n_entry_fund": statistics.fmean(r["n_entry_fund"] for r in rows),
        "n_exit_fund": statistics.fmean(r["n_exit_fund"] for r in rows),
        "n_common_arb": statistics.fmean(r["n_common_arb"] for r in rows),
        "n_entry_arb": statistics.fmean(r["n_entry_arb"] for r in rows),
        "n_exit_arb": statistics.fmean(r["n_exit_arb"] for r in rows),
    },
    "components": stats,
    "main_table": table_rows,
}

RESULT_PATH.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
print(f"wrote {RESULT_PATH}")
print(f"delta_A total mean = {total_mean:.1f}")
for tr in table_rows:
    share = tr["share_of_delta_a_pct"]
    share_s = f"{share:.1f}%" if share is not None else "n/a"
    print(f"  {tr['component']:24s} {tr['estimate']:>10,d}  share={share_s}")
