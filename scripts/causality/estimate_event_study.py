#!/usr/bin/env python3
"""Primary matched-overlap event-study (Eq. 3) on the VALIDATED panel, using canonical
Python estimators (pyfixest for two-way FE + cluster-robust; wildboottest for WCR). This is
the paper-reporting side; the Rust `event_study` binary reproduces it as a parity check.

DISCIPLINE: prints PRE-TREND (lead) coefficients first; POST (lag) coefficients are labelled
PROVISIONAL and must not be interpreted / put in the manuscript until the pre-trend is
reviewed. No ATT is asserted here.

Design (mirrors src/causal/adapter.rs + design_meta.rs exactly):
- matched-overlap sample from matched_pairs.json: matched treated (weight 1) + matched
  controls (weight = with-replacement multiplicity); unmatched treated / unused controls excluded
- cluster = token pair (sorted token0,token1 from ckpt_tokens.json)
- event time = frozen-grid index(week) - index(t0); t0 = fee-switch week (default 2025-51)
- frozen window 2024-01-01..2026-06-30; event time trimmed to +/-HORIZON (default 12)
- outcome default twl_active_liquidity (decimals-independent)

Usage: estimate_event_study.py [PANEL_CSV] [--outcome NAME] [--t0 2025-51] [--horizon 12]
                               [--reps 9999] [--out DIR]
"""
import argparse, csv, json, os, calendar, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

ROOT = os.environ.get("AMMLAB_ROOT", "/Users/joseph/amm-lab")
DATA = os.environ.get("AMMLAB_DATA", os.path.join(ROOT, ".local/amm_paper_c/data"))


def frozen_grid_index():
    """{'%Y-%W': contiguous index} over 2024-01-01..2026-06-30, matching the Rust WeekGrid."""
    b0 = calendar.timegm((2024, 1, 1, 0, 0, 0))
    b1 = calendar.timegm((2026, 6, 30, 23, 59, 59))
    labels = set()
    t = b0
    while t <= b1:
        labels.add(time.strftime("%Y-%W", time.gmtime(t)))
        t += 86400
    return {lab: i for i, lab in enumerate(sorted(labels))}  # string sort == chronological for %Y-%W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", nargs="?", default=os.path.join(DATA, "panel_weekly_rust.csv"))
    ap.add_argument("--outcome", default="twl_active_liquidity")
    # Liquidity-magnitude outcomes span ~20 orders of magnitude in levels -> FE demeaning is
    # ill-conditioned and whale-dominated. asinh (log-like, handles 0 / negatives) is the
    # default; `none` = raw levels; `log1p` for strictly-nonneg outcomes. This transform is a
    # DESIGN CHOICE and must be pinned in the frozen PAP.
    ap.add_argument("--transform", default="asinh", choices=["asinh", "log1p", "none"])
    ap.add_argument("--t0", default="2025-51")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--reps", type=int, default=9999)
    ap.add_argument("--out", default=os.path.join(DATA, "event_study_py_out"))
    a = ap.parse_args()

    grid = frozen_grid_index()
    if a.t0 not in grid:
        raise SystemExit(f"t0 {a.t0} not in frozen grid")
    t0i = grid[a.t0]

    # ---- design metadata ----
    tokens = json.load(open(os.path.join(DATA, "ckpt_tokens.json")))
    cluster_of = {}
    for pool, pair in tokens.items():
        ts = sorted(x.lower() for x in pair if x)
        cluster_of[pool] = "-".join(ts) if ts else "na"

    mp = json.load(open(os.path.join(DATA, "matched_pairs.json")))
    treated = {m["treated"] for m in mp}
    freq = Counter()
    for m in mp:
        for c in m.get("controls", []):
            freq[c] += 1
    weight_of = {t: 1.0 for t in treated}
    for c, f in freq.items():
        weight_of[c] = float(f)  # control multiplicity as frequency weight

    # ---- panel -> analysis frame ----
    df = pd.read_csv(a.panel)
    df = df[df["week"].isin(grid)].copy()  # frozen window only (drops 2023-52 boundary)
    df = df[df["pool"].isin(weight_of)].copy()  # matched-overlap units only
    df["treated"] = df["pool"].isin(treated).astype(int)
    df["w"] = df["pool"].map(weight_of)
    df["cluster_key"] = df["pool"].map(cluster_of).fillna("na")
    df["widx"] = df["week"].map(grid)
    df["rel"] = df["widx"] - t0i
    # trim to the estimation horizon (windowed event study; endpoints dropped, not binned)
    df = df[df["rel"].between(-a.horizon, a.horizon)].copy()
    # role consistency guard (mirror the Rust adapter)
    bad = df[(df["treated"] == 1) != df["pool"].isin(treated)]
    assert bad.empty, "role/treated inconsistency"

    y = a.outcome
    if y not in df.columns:
        raise SystemExit(f"unknown outcome {y}; columns: {list(df.columns)}")
    df = df.dropna(subset=[y])
    # outcome transform (see --transform note): compress the huge dynamic range for FE demeaning
    ycol = y
    if a.transform == "asinh":
        df["_yt"] = np.arcsinh(df[y].astype(float)); ycol = "_yt"
    elif a.transform == "log1p":
        df["_yt"] = np.log1p(df[y].clip(lower=0).astype(float)); ycol = "_yt"
    n_treated = df.loc[df.treated == 1, "pool"].nunique()
    n_control = df.loc[df.treated == 0, "pool"].nunique()
    print(f"matched-overlap: {n_treated} treated + {n_control} control pools | "
          f"{len(df)} pool-weeks | clusters {df.cluster_key.nunique()} | "
          f"rel in [-{a.horizon},{a.horizon}] | outcome {y} [{a.transform}] | t0 {a.t0}", flush=True)

    import pyfixest as pf
    # event-study interaction: treated x relative-time dummies, ref = -1; pool + week FE absorbed
    fml = f"{ycol} ~ i(rel, treated, ref=-1) | pool + week"
    m = pf.feols(fml, data=df, weights="w", vcov={"CRV1": "cluster_key"})
    tab = m.tidy()  # Estimate, Std. Error, t, Pr(>|t|), 2.5%, 97.5% indexed by coef name

    # parse rel from coef names like "rel::0:treated" / "C(rel)[T.3]:treated"
    def rel_of(name):
        import re
        mm = re.search(r"rel[^-\d]*(-?\d+)", name)
        return int(mm.group(1)) if mm else None

    rows = []
    for name, r in tab.iterrows():
        k = rel_of(name)
        if k is None:
            continue
        rows.append({"rel": k, "coef": name,
                     "beta": r["Estimate"], "se": r["Std. Error"],
                     "ci_lo": r.get("2.5%", np.nan), "ci_hi": r.get("97.5%", np.nan),
                     "crv1_p": r["Pr(>|t|)"]})
    rows.sort(key=lambda x: x["rel"])

    # ---- joint pre-trend tests from the cluster-robust vcov ----
    # NOTE: WCR is NOT computed here -- pyfixest's wild bootstrap does not support WLS
    # (our matched-control frequency weights). The reported WCR comes from the Rust
    # `event_study` estimator (weighted design via sqrt(w)); Python supplies CRV1 + these
    # joint tests, and the point estimates cross-check against Rust (parity).
    from scipy.stats import chi2, norm
    names = list(m._coefnames)
    beta = np.asarray(m.coef().reindex(names), dtype=float)
    V = np.asarray(m._vcov, dtype=float)
    lead_ix = [i for i, n in enumerate(names) if (rel_of(n) is not None and rel_of(n) < 0)]
    joint_p = slope_p = slope = max_abs_pre = float("nan")
    if lead_ix:
        bL = beta[lead_ix]
        VL = V[np.ix_(lead_ix, lead_ix)]
        # joint Wald: all leads = 0  (H0: parallel pre-trends)
        W = float(bL @ np.linalg.pinv(VL) @ bL)
        joint_p = float(chi2.sf(W, len(lead_ix)))
        # lead linear-trend contrast: systematic slope across the leads
        rels = np.array([rel_of(names[i]) for i in lead_ix], dtype=float)
        g = rels - rels.mean()
        gamma = float(g @ bL)
        gvar = float(g @ VL @ g)
        if gvar > 0:
            z = gamma / np.sqrt(gvar)
            slope = gamma
            slope_p = float(2 * norm.sf(abs(z)))
        max_abs_pre = float(np.max(np.abs(bL)))

    n_leads = len(lead_ix)
    summary = {
        "outcome": y, "transform": a.transform, "t0": a.t0, "horizon": a.horizon,
        "matched_treated": int(n_treated), "matched_control": int(n_control),
        "pool_weeks": int(len(df)), "clusters": int(df.cluster_key.nunique()),
        "n_leads": n_leads, "n_lags": sum(1 for r in rows if r["rel"] >= 0),
        "joint_leads_p": joint_p, "lead_slope": slope, "lead_slope_p": slope_p,
        "max_abs_pre_coef": max_abs_pre,
        "wcr_source": "rust event_study estimator (pyfixest cannot WCR under WLS)",
    }

    os.makedirs(a.out, exist_ok=True)
    outcsv = os.path.join(a.out, f"event_study_{y}.csv")
    with open(outcsv, "w", newline="") as g_:
        w = csv.DictWriter(g_, fieldnames=["rel", "beta", "se", "ci_lo", "ci_hi", "crv1_p"])
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in ["rel", "beta", "se", "ci_lo", "ci_hi", "crv1_p"]})
    json.dump(summary, open(os.path.join(a.out, f"summary_{y}.json"), "w"), indent=1)

    # export event-study beta + cluster-robust vcov for HonestDiD (leads then lags, ref -1
    # omitted): betahat ordered by rel, numPrePeriods = #leads, numPostPeriods = #lags.
    es_ix = [i for i, n in enumerate(names) if rel_of(n) is not None]
    es_ix.sort(key=lambda i: rel_of(names[i]))
    es_rel = [rel_of(names[i]) for i in es_ix]
    es_beta = [float(beta[i]) for i in es_ix]
    es_vcov = [[float(V[i, j]) for j in es_ix] for i in es_ix]
    json.dump({"outcome": y, "transform": a.transform, "t0": a.t0,
               "rel": es_rel, "beta": es_beta, "vcov": es_vcov,
               "num_pre": sum(1 for r in es_rel if r < 0),
               "num_post": sum(1 for r in es_rel if r >= 0)},
              open(os.path.join(a.out, f"es_beta_vcov_{y}.json"), "w"))
    print(f"wrote {outcsv} + summary_{y}.json + es_beta_vcov_{y}.json", flush=True)

    def show(sel, title):
        print(f"\n=== {title} ===")
        print(f"  {'rel':>4} {'beta':>16} {'se':>14} {'crv1_p':>9}")
        for row in rows:
            if sel(row["rel"]):
                print(f"  {row['rel']:>4} {row['beta']:>16.4f} {row['se']:>14.4f} {row['crv1_p']:>9.4f}")

    show(lambda k: k < 0, "PRE-TREND (leads, rel<0; rel=-1 omitted as reference)")
    print("\n=== POST (lags, rel>=0) -- PROVISIONAL: do NOT interpret until pre-trend reviewed ===")
    show(lambda k: k >= 0, "POST (lags)")

    print("\n=== PRE-TREND SUMMARY (the decision-relevant numbers) ===")
    print(f"  outcome/transform      : {y} [{a.transform}]")
    print(f"  matched treated/control: {n_treated} / {n_control}")
    print(f"  pool-weeks / clusters  : {len(df)} / {df.cluster_key.nunique()}")
    print(f"  leads / lags           : {n_leads} / {summary['n_lags']}")
    print(f"  JOINT leads=0 (Wald) p : {joint_p:.4f}   <- overall parallel-pre-trend test")
    print(f"  lead linear-slope p    : {slope_p:.4f}   (slope={slope:+.4f} per week)")
    print(f"  max |pre-coef|         : {max_abs_pre:.4f}")
    print(f"  WCR p-values           : from Rust estimator (pyfixest WCR unavailable under WLS); CRV1 shown above")


if __name__ == "__main__":
    main()
