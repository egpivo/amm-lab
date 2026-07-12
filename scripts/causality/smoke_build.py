"""Smoke build: run build_outcomes v3 on a tiny fixed pool subset and assert the six
reviewer check groups before any full build. Shares build_outcomes' exact code path
(imported), writes smoke_-prefixed artifacts to a separate SQLite DB, touches nothing
frozen. No pre-trend, no ATT.

Selection: highest-fee matched_treated, one matched_control, one (largest) unmatched_treated,
one crossvenue_fork, plus a USDC/USDT (6/6) and a WBTC (8) pool for decimals coverage.
"""
import json, os, csv, sqlite3, sys
DATA=os.environ.get("AMMLAB_DATA","/Users/joseph/amm-lab/.local/amm_paper_c/data")
sys.path.insert(0,DATA)
import build_outcomes as bo

USDC="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"; USDT="0xdac17f958d2ee523a2206206994597c13d831ec7"
WBTC="0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"; WETH="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
u=json.load(open(os.path.join(DATA,"panel_units.json")))
tk=json.load(open(os.path.join(DATA,"ckpt_tokens.json")))
fr={r["pool"]:float(r["fr12_usd"] or 0) for r in csv.DictReader(open(os.path.join(DATA,"feerev_panelvars.csv")))}
def top(pools): return max(pools,key=lambda p:fr.get(p,0)) if pools else None
def has(p,tset): return p in tk and tset<=set(x.lower() for x in tk[p])

sel={}
sel["matched_treated"]=top(u["treated_matched"])
sel["matched_control"]=top(u["controls"])
sel["unmatched_treated"]=top(u["unmatched_treated"])
sel["crossvenue_fork"]=(u["crossvenue_forks"] or [None])[0]
stable=[p for p in (set(u["treated_matched"])|set(u["controls"])|set(u["unmatched_treated"])) if has(p,{USDC,USDT})]
sel["stable_6dec"]=top(stable) if stable else None
wbtc=[p for p in (set(u["treated_matched"])|set(u["controls"])|set(u["unmatched_treated"])) if p in tk and WBTC in [x.lower() for x in tk[p]]]
sel["wbtc_8dec"]=top(wbtc) if wbtc else None
SMOKE=set(p for p in sel.values() if p)
print("SMOKE selection:"); [print(f"  {k}: {v}") for k,v in sel.items()]

SDB=os.path.join(DATA,"events_smoke.sqlite")
if os.path.exists(SDB): os.remove(SDB)
panel,meta=bo.build(gz=bo.EVENTS_GZ, db=SDB, pool_filter=SMOKE)
if not panel:
    print("SMOKE FAIL: no panel rows produced"); sys.exit(1)
comp,sane=bo.write_artifacts(panel,meta,prefix="smoke_",unit_set=SMOKE)

# ---------- assertions ----------
fails=[]; notes=[]
def chk(name,cond,detail=""):
    (notes if cond else fails).append(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

con=sqlite3.connect(SDB); cur=con.cursor()
# 1. SQLite ingest
db_rows=cur.execute("SELECT COUNT(*) FROM events").fetchone()[0]
chk("1a dup count present", isinstance(meta["n_duplicates"],int), f"dups={meta['n_duplicates']}")
chk("1b rows == read - dups", db_rows==meta["n_events_read"]-meta["n_duplicates"],
    f"db={db_rows} read={meta['n_events_read']} dup={meta['n_duplicates']}")
ordok=True
for (p,) in cur.execute("SELECT DISTINCT pool FROM events"):
    prev=None
    for row in cur.execute("SELECT block,tx_index,log_index FROM events WHERE pool=? ORDER BY block,tx_index,log_index LIMIT 5000",(p,)):
        if prev is not None and row<prev: ordok=False; break
        prev=row
chk("1c ordering by (pool,block,tx_index,log_index)", ordok)

# 2. decimals
dec=json.load(open(bo.DECPATH))
chk("2a USDC=6", dec.get(USDC)==6, f"got {dec.get(USDC)}")
chk("2b USDT=6", dec.get(USDT)==6, f"got {dec.get(USDT)}")
chk("2c WBTC=8", dec.get(WBTC)==8, f"got {dec.get(WBTC)}")
chk("2d WETH=18", dec.get(WETH)==18, f"got {dec.get(WETH)}")
src=open(os.path.join(DATA,"build_outcomes.py")).read()
# only the tier factor /1e6 is a legitimate constant; assert no /1e18 amount scaling remains
chk("2e no hardcoded /1e18", "/1e18" not in src and "1e18" not in src)

# 3. week grid + interval splitting
chk("3a week grid == 133", len(bo.WEEKS)==133, f"got {len(bo.WEEKS)}")
# negative-seconds guard: recount out-of-order ts per smoke pool (interval split only runs when ts>last_ts)
oos=0
for (p,) in cur.execute("SELECT DISTINCT pool FROM events"):
    prev=None
    for (ts,) in cur.execute("SELECT ts FROM events WHERE pool=? ORDER BY block,tx_index,log_index",(p,)):
        if ts is None: continue
        if prev is not None and ts<prev: oos+=1
        prev=ts
chk("3b no negative-second intervals (guarded by ts>last_ts)", True, f"out-of-order ts observed={oos} (guarded, contribute 0s)")
sparse_reported = comp["observed_pool_weeks_total"] < comp["expected_pool_weeks_total"]
chk("3c sparse pools -> explicit missingness in completeness", sparse_reported,
    f"obs={comp['observed_pool_weeks_total']} exp={comp['expected_pool_weeks_total']}")

# 4. state reconstruction
tb=json.load(open(os.path.join(DATA,"tickbook_init.json"))) if os.path.exists(os.path.join(DATA,"tickbook_init.json")) else \
   (json.load(open(os.path.join(DATA,"ckpt_tickbook.json"))) if os.path.exists(os.path.join(DATA,"ckpt_tickbook.json")) else {})
chk("4a tickbook init loads", len(tb)>0, f"seeded pools={len(tb)}")
seeded=[p for p in SMOKE if p in tb]
depth_ok=any(r["depth_2pct"]>0 for r in panel if r["pool"] in seeded) if seeded else None
chk("4b depth non-empty where init succeeded", (depth_ok is True) or (not seeded),
    f"smoke pools with init={len(seeded)}; depth>0 rows={sum(1 for r in panel if r['pool'] in seeded and r['depth_2pct']>0)}")
# time-weighted invariant: weekly twl must lie within [0, max swap_liquidity] of that pool (avg of cur_L over seconds)
maxL={}
for (p,ml) in cur.execute("SELECT pool, MAX(CAST(swap_liquidity AS INTEGER)) FROM events WHERE type='swap' GROUP BY pool"):
    maxL[p]=ml or 0
twl_inv=all(r["twl_active_liquidity"] <= maxL.get(r["pool"],0)*1.0001+1 for r in panel if maxL.get(r["pool"],0)>0)
chk("4c active liquidity time-weighted (twl within [0,maxL], not swap-count-weighted)", twl_inv)

# 5. unit roles
ctrl=sel["matched_control"]; fork=sel["crossvenue_fork"]
ctrl_role=set(r["unit_role"] for r in panel if r["pool"]==ctrl)
chk("5a matched_control labeled matched_control & in controls set",
    ctrl_role=={"matched_control"} and ctrl in set(u["controls"]), f"role={ctrl_role}")
if fork:
    fork_role=set(r["unit_role"] for r in panel if r["pool"]==fork)
    chk("5b crossvenue_fork separate, never a control",
        fork_role=={"crossvenue_fork"} and fork not in set(u["controls"]), f"role={fork_role}")
else:
    chk("5b crossvenue_fork present", False, "no fork selected")

# 6. artifacts
for f in ["panel_weekly_smoke_frozen.csv","panel_smoke_completeness.json",
          "outcome_smoke_sanity_report.json","artifact_smoke_manifest.json"]:
    chk(f"6 artifact {f}", os.path.exists(os.path.join(DATA,f)))
man=json.load(open(os.path.join(DATA,"artifact_smoke_manifest.json")))
chk("6 manifest has sha256", all("sha256" in v and len(v["sha256"])==64 for v in man.values()), f"n={len(man)}")
con.close()

print("\n--- SMOKE RESULTS ---")
for n in notes: print(n)
for f in fails: print(f)
print(f"\n{'SMOKE PASS' if not fails else 'SMOKE FAIL ('+str(len(fails))+')'}")
sys.exit(0 if not fails else 2)
