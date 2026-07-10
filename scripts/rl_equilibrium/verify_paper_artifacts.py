"""Verify every number the paper's tables and headline claims rest on,
directly from the artifact CSVs. Exits non-zero on any mismatch.

Checks:
1. Final block (Table 1): means for DQN/lookahead/TWAP per ordering,
   completion == 1.0 on every row, seed range 90000-90999, n = 1000.
2. Final ladder (Figure 1): means per policy.
3. Stochastic planner final tie.
4. No duplicate (policy, seed[, cell]) rows in append-style CSVs.
5. M4 sensitivity: completion 1.0, paired-edge sign never flips.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "experiments/rl_execution/out"
TOL = 0.05  # bps tolerance on recomputed means
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def mean(rows, key="shortfall_bps"):
    vals = [float(r[key]) for r in rows]
    return sum(vals) / len(vals)


def main() -> None:
    # ---- 1. Table 1: final block ----
    ref = list(csv.DictReader(open(OUT / "m3r_reference_final.csv")))
    dqn = list(csv.DictReader(open(OUT / "m3r_final_paper_seeds.csv")))
    expected = {  # (policy_file, ordering) -> paper mean
        ("dqn_dynamic_duopoly", "before"): 85.73,
        ("dqn_order_random", "random"): 96.56,
        ("dqn_order_after", "after"): 100.32,
    }
    la_expected = {"before": 100.63, "random": 102.18, "after": 113.61}
    tw_expected = {"before": 111.24, "random": 112.92, "after": 123.12}
    for (pol, order), want in expected.items():
        rows = [r for r in dqn if r["policy"] == pol and r["agent_order"] == order]
        check(f"table1 {pol}/{order} n=1000", len(rows) == 1000, f"n={len(rows)}")
        seeds = sorted(int(r["seed"]) for r in rows)
        check(f"table1 {pol}/{order} seed block", seeds[0] == 90_000 and seeds[-1] == 90_999)
        got = mean(rows)
        check(f"table1 {pol}/{order} mean {want}", abs(got - want) < TOL, f"got {got:.2f}")
        comp = min(float(r["completion_rate"]) for r in rows)
        check(f"table1 {pol}/{order} completion==1.0", comp > 1 - 1e-9)
    for order, want in la_expected.items():
        rows = [r for r in ref if r["policy"] == "lookahead" and r["agent_order"] == order
                and r["mode"] == "dynamic_duopoly"]
        got = mean(rows)
        check(f"table1 lookahead/{order} mean {want}", abs(got - want) < TOL, f"got {got:.2f}")
    for order, want in tw_expected.items():
        rows = [r for r in ref if r["policy"] == "twap" and r["agent_order"] == order
                and r["mode"] == "dynamic_duopoly"]
        got = mean(rows)
        check(f"table1 twap/{order} mean {want}", abs(got - want) < TOL, f"got {got:.2f}")

    # ---- 2. Figure 1: final ladder ----
    ladder = list(csv.DictReader(open(OUT / "final_ladder.csv")))
    fig1 = {"twap": 111.24, "lookahead": 100.63, "two_step": 111.45,
            "three_step": 102.25, "q_learner": 108.86, "q_learner_fine": 100.21,
            "clairvoyant": 78.20}
    for pol, want in fig1.items():
        rows = [r for r in ladder if r["policy"] == pol]
        got = mean(rows)
        check(f"fig1 {pol} mean {want}", len(rows) == 1000 and abs(got - want) < TOL,
              f"got {got:.2f}, n={len(rows)}")

    # ---- 3. Stochastic planner tie (corrected model) ----
    sp = list(csv.DictReader(open(OUT / "m3r_stochastic_planner_final.csv")))
    by = defaultdict(dict)
    for r in sp:
        by[r["policy"]][r["seed"]] = float(r["shortfall_bps"])
    diffs = [by["stochastic_planner"][s] - by["lookahead"][s] for s in by["stochastic_planner"]]
    m = sum(diffs) / len(diffs)
    check("planner final tie |paired| < 0.7 bps", abs(m) < 0.7, f"paired {m:+.2f}")

    # ---- 4. No duplicate rows in append-style CSVs ----
    for fname, keycols in [
        ("m3r_completion.csv", ("policy", "completion_rule", "seed_set", "seed")),
        ("m4_lp_adaptation.csv", ("extension", "regime", "policy", "seed")),
        ("m4_jit_mev.csv", ("extension", "regime", "policy", "seed")),
    ]:
        rows = list(csv.DictReader(open(OUT / fname)))
        keys = Counter(tuple(r[c] for c in keycols) for r in rows)
        dups = [k for k, n in keys.items() if n > 1]
        check(f"no duplicate rows in {fname}", not dups,
              f"{len(dups)} duplicated keys" if dups else "")

    # ---- 5. M4: completion 1.0 and edge sign stable ----
    for fname in ["m4_lp_adaptation.csv", "m4_jit_mev.csv"]:
        rows = list(csv.DictReader(open(OUT / fname)))
        comp = min(float(r["completion_rate"]) for r in rows)
        check(f"{fname} completion==1.0", comp > 1 - 1e-9)
        cells = defaultdict(lambda: defaultdict(dict))
        for r in rows:
            cells[r["regime"]][r["policy"]][r["seed"]] = float(r["shortfall_bps"])
        for regime, pols in cells.items():
            d = [pols["dqn"][s] - pols["lookahead"][s] for s in pols["dqn"]]
            edge = sum(d) / len(d)
            check(f"{fname} {regime} DQN edge negative", edge < 0, f"{edge:+.2f}")

    print(f"\n{len(FAILURES)} failure(s)")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
