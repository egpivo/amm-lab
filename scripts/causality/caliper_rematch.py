#!/usr/bin/env python3
"""Caliper sensitivity re-match (Amendment 016). Regenerates the matched-overlap design
artifact at a given caliper using the EXACT frozen matching procedure from
recompute_feasibility.py -- only the caliper threshold changes -- so the 0.25/0.5/1.0
comparison isolates the overlap definition, not a matcher re-implementation. NO RPC (uses the
frozen ckpt_factory.json as-is). Pre-period / design-only.

Frozen procedure held fixed: control reservoir = untreated & old & covered & fr12>0 &
factory==CANON & exposure<=0.25 (low-exposure); exact (tier,class) strata; <=3 nearest
neighbours by |log(fr12)| distance; caliper on absolute log-fee-revenue.

Usage: caliper_rematch.py --caliper 0.25   (writes matched_pairs_cal0.25.json + stats)
"""
import csv, json, math, os, argparse, statistics as st
from collections import defaultdict, Counter

ROOT = os.environ.get("AMMLAB_ROOT", "/Users/joseph/amm-lab")
DATA = os.environ.get("AMMLAB_DATA", os.path.join(ROOT, ".local/amm_paper_c/data"))
CANON = "0x1f98431c8ad98523631ae4a59f267346ea31f984"

ap = argparse.ArgumentParser()
ap.add_argument("--caliper", type=float, required=True)
a = ap.parse_args()
C = a.caliper

rows = {r["pool"]: r for r in csv.DictReader(open(os.path.join(DATA, "feerev_panelvars.csv")))}
tok = json.load(open(os.path.join(DATA, "ckpt_tokens.json")))
fac = json.load(open(os.path.join(DATA, "ckpt_factory.json")))
treated = {p for p, r in rows.items() if r["treated"] == "1"}
reservoir = [p for p, r in rows.items()
             if r["treated"] == "0" and r["old"] == "1" and r["covered"] == "1" and float(r["fr12_usd"]) > 0]

tpairs, ttok = set(), set()
for p in treated:
    t = tok.get(p, [None, None])
    if t[0]:
        tpairs.add(frozenset(t)); ttok.update(t)
def exp(p):
    t = tok.get(p, [None, None])
    if not t[0]:
        return 1.0
    if frozenset(t) in tpairs:
        return 1.0
    if t[0] in ttok or t[1] in ttok:
        return 0.125
    return 0.0

canon_reservoir = [p for p in reservoir if fac.get(p) == CANON and exp(p) <= 0.25]
cands = defaultdict(list)
for p in canon_reservoir:
    r = rows[p]; cands[(r["tier"], r["class"])].append(p)

def lfr(p): return math.log(float(rows[p]["fr12_usd"]))
treated_main = [p for p, r in rows.items()
                if r["treated"] == "1" and r["old"] == "1" and r["covered"] == "1" and float(r["fr12_usd"]) > 0]

matched, unmatched = [], []
for p in treated_main:
    r = rows[p]; pool = cands.get((r["tier"], r["class"]), []); lg = lfr(p)
    near = sorted(pool, key=lambda c: abs(lfr(c) - lg))[:3]
    near = [c for c in near if abs(lfr(c) - lg) <= C]                 # <-- only the caliper varies
    if near:
        matched.append({"treated": p, "fr12": float(r["fr12_usd"]), "tier": r["tier"], "class": r["class"], "controls": near})
    else:
        unmatched.append({"pool": p, "fr12": float(r["fr12_usd"]), "tier": r["tier"], "class": r["class"]})

ctrl_used = sorted({c for m in matched for c in m["controls"]})
reuse = Counter(c for m in matched for c in m["controls"])
tm = [m["treated"] for m in matched]

def smd(A, B, f):
    xa = [f(x) for x in A]; xb = [f(x) for x in B]
    sp = math.sqrt((st.pstdev(xa) ** 2 + st.pstdev(xb) ** 2) / 2) or 1e-9
    return (st.mean(xa) - st.mean(xb)) / sp

tag = f"cal{C:g}"
json.dump(matched, open(os.path.join(DATA, f"matched_pairs_{tag}.json"), "w"), indent=1)
stats = {
    "caliper": C,
    "n_treated_main": len(treated_main),
    "n_matched": len(matched),
    "match_rate": round(len(matched) / len(treated_main), 4),
    "n_candidate_controls": len(canon_reservoir),
    "n_controls_used": len(ctrl_used),
    "n_unmatched": len(unmatched),
    "control_reuse_max": max(reuse.values()) if reuse else 0,
    "control_reuse_mean": round(sum(reuse.values()) / len(reuse), 3) if reuse else 0,
    "smd_logfr_before": round(smd(tm, canon_reservoir, lfr), 4) if tm else None,
    "smd_logfr_after": round(smd(tm, ctrl_used, lfr), 4) if ctrl_used else None,
}
json.dump(stats, open(os.path.join(DATA, f"match_stats_{tag}.json"), "w"), indent=1)
print(f"caliper {C}: matched {stats['n_matched']}/{stats['n_treated_main']} "
      f"(rate {stats['match_rate']}), controls used {stats['n_controls_used']}, "
      f"unmatched {stats['n_unmatched']}, reuse max {stats['control_reuse_max']}, "
      f"SMD logfr {stats['smd_logfr_before']}->{stats['smd_logfr_after']}", flush=True)
