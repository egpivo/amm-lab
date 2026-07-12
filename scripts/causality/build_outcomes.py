"""Task 2b stage 2 (v3): reconstruct the frozen primary outcome family.

Patches over v2 (reviewer W-1..W-3):
  W-1 decimals: scale amount0/amount1/Collect/volume/fee-income by per-token decimals
      (from token_decimals.json, RPC-filled at build time). No hardcoded /1e18.
  W-2 memory: do NOT load all events into RAM. Ingest events.csv.gz into SQLite once,
      index by pool, then reconstruct one pool at a time ordered by
      (pool, block, tx_index, log_index). Weekly aggregates (small) stay in RAM.
  W-3 completeness: expected pool-weeks = frozen unit set x frozen week grid; report
      observed-vs-expected by unit_role and by outcome; include failed RPC ranges,
      incomplete tickbook-init pools, missing outcomes, duplicate counts, artifact SHA256.

Depth uses a bisect-cached running tick book seeded from tickbook_init.json (BO-2).
No pre-trend, no ATT. Emits panel_weekly_frozen.csv/.parquet, panel_completeness.json,
outcome_sanity_report.json, state_initialization_report.json, artifact_manifest.json.

NOTE (fee income units): fee_income and its per-liquidity ratio are in NATIVE token1
units (|amount1_human| x fee_tier). Cross-pool USD comparability requires a weekly price
map and is deliberately deferred; this is flagged in outcome_sanity_report.json so no
downstream estimate silently treats native-unit fee income as USD.
"""
import json, os, csv, gzip, math, time, hashlib, calendar, bisect, sqlite3, urllib.request
from collections import defaultdict, Counter
ROOT=os.environ.get("AMMLAB_ROOT","/Users/joseph/amm-lab")
DATA=os.environ.get("AMMLAB_DATA", os.path.join(ROOT,".local/amm_paper_c/data"))
EVENTS_GZ=os.path.join(DATA,"events","events.csv.gz")
DB=os.path.join(DATA,"events_sorted.sqlite")
B0=calendar.timegm((2024,1,1,0,0,0)); B1=calendar.timegm((2026,6,30,23,59,59))
SECWK=7*86400

def week_id(ts): return time.strftime("%Y-%W", time.gmtime(ts))
def week_start(ts):
    tm=time.gmtime(ts)
    y,w=int(time.strftime("%Y",tm)),int(time.strftime("%W",tm))
    return calendar.timegm(time.strptime(f"{y} {w} 1","%Y %W %w"))
def week_grid():
    ws=set(); t=B0
    while t<=B1: ws.add(week_id(t)); t+=86400
    return ws
WEEKS=week_grid()

# ---- W-1: token decimals map (RPC-filled at build time, cached) --------------
def rpcurl():
    u=os.environ.get("ALCHEMY_ETHEREUM_URL")
    if u: return u.strip().strip('"')
    for l in open(os.path.join(ROOT,".env")):
        if l.startswith("ALCHEMY_ETHEREUM_URL"): return l.split("=",1)[1].strip().strip('"')
RPC=rpcurl()
def rpc(m,p,retries=5):
    d=json.dumps({"jsonrpc":"2.0","id":1,"method":m,"params":p}).encode()
    for a in range(retries):
        try:
            r=urllib.request.Request(RPC,data=d,headers={"Content-Type":"application/json"})
            o=json.loads(urllib.request.urlopen(r,timeout=45).read())
            if "error" in o: raise RuntimeError(str(o["error"])[:120])
            return o["result"]
        except Exception:
            if a==retries-1: raise
            time.sleep(1.0*(a+1))
DECPATH=os.path.join(DATA,"token_decimals.json")
def load_decimals(tokens):
    dec=json.load(open(DECPATH)) if os.path.exists(DECPATH) else {}
    dec_default=[]  # tokens that failed -> default 18, flagged
    miss=[t for t in tokens if t and t not in dec]
    for i,t in enumerate(miss):
        try:
            res=rpc("eth_call",[{"to":t,"data":"0x313ce567"},"latest"])  # decimals()
            dec[t]=int(res,16) if res and res!="0x" else 18
            if not res or res=="0x": dec_default.append(t)
        except Exception:
            dec[t]=18; dec_default.append(t)
        if i%100==0:
            json.dump(dec,open(DECPATH,"w")); print(f"  decimals {i}/{len(miss)}",flush=True)
    json.dump(dec,open(DECPATH,"w"))
    return dec, dec_default

# ---- W-2: ingest events.csv.gz into SQLite once, index by pool ---------------
def ingest(gz=EVENTS_GZ, db=DB, pool_filter=None):
    n_read=0; n_ins=0
    con=sqlite3.connect(db); cur=con.cursor()
    cur.execute("DROP TABLE IF EXISTS events")
    cur.execute("""CREATE TABLE events(
        pool TEXT, unit_role TEXT, tx_hash TEXT, block INTEGER, tx_index INTEGER,
        log_index INTEGER, ts INTEGER, type TEXT, owner TEXT, tickLower INTEGER,
        tickUpper INTEGER, liquidity_delta TEXT, swap_liquidity TEXT, amount0 TEXT,
        amount1 TEXT, sqrtP TEXT, tick INTEGER, token0 TEXT, token1 TEXT, removed TEXT,
        UNIQUE(tx_hash, log_index))""")
    con.commit()
    def val(r):
        def gi(k):
            try: return int(r[k]) if r[k] not in ("",None) else None
            except: return None
        return (r["pool"],r["unit_role"],r["tx_hash"],gi("block"),gi("tx_index"),gi("log_index"),
                gi("ts"),r["type"],r.get("owner",""),gi("tickLower"),gi("tickUpper"),
                r.get("liquidity_delta","0"),r.get("swap_liquidity","0"),r.get("amount0","0"),
                r.get("amount1","0"),r.get("sqrtP","0"),gi("tick"),r.get("token0",""),
                r.get("token1",""),r.get("removed",""))
    batch=[]
    for r in csv.DictReader(gzip.open(gz,"rt")):
        if pool_filter is not None and r["pool"] not in pool_filter: continue
        n_read+=1; batch.append(val(r))
        if len(batch)>=50000:
            cur.executemany("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",batch)
            n_ins+=cur.rowcount if cur.rowcount>0 else 0; con.commit(); batch=[]
    if batch:
        cur.executemany("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",batch)
        con.commit()
    cur.execute("CREATE INDEX ix_pool ON events(pool, block, tx_index, log_index)")
    con.commit()
    n_final=cur.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    con.close()
    return n_read, n_final, n_read-n_final  # read, inserted(unique), duplicates

# ---- depth: bisect-cached running tick book ----------------------------------
def bt(pct): return int(math.log(1+pct)/math.log(1.0001))
BANDS={"1pct":bt(0.01),"2pct":bt(0.02),"5pct":bt(0.05)}
class Book:
    def __init__(self): self.net=defaultdict(int); self._dirty=True; self._ticks=[]; self._cum=[]
    def apply(self,tick,dL): self.net[tick]+=dL; self._dirty=True
    def _rebuild(self):
        self._ticks=sorted(t for t,v in self.net.items() if v!=0); c=[]; s=0
        for t in self._ticks: s+=self.net[t]; c.append(s)
        self._cum=c; self._dirty=False
    def active_L(self,tick):
        if self._dirty: self._rebuild()
        if not self._ticks: return 0
        i=bisect.bisect_right(self._ticks,tick)
        return max(self._cum[i-1],0) if i>0 else 0

# ---- frozen unit set & roles (W-3 expected) ----------------------------------
units=json.load(open(os.path.join(DATA,"panel_units.json")))
role_of={}
for p in units.get("treated_matched",[]): role_of[p]="matched_treated"
for p in units.get("controls",[]): role_of[p]="matched_control"
for p in units.get("unmatched_treated",[]): role_of[p]="unmatched_treated"
for p in units.get("crossvenue_forks",[]): role_of[p]="crossvenue_fork"
UNIT_SET=set(role_of)
tier={r["pool"]:int(r["tier"]) for r in csv.DictReader(open(os.path.join(DATA,"feerev_panelvars.csv"))) if r["tier"] not in ("","0")}
tickbook=json.load(open(os.path.join(DATA,"tickbook_init.json"))) if os.path.exists(os.path.join(DATA,"tickbook_init.json")) else {}

def build(gz=EVENTS_GZ, db=DB, pool_filter=None):
    print("ingesting events -> sqlite (out-of-core)...",flush=True)
    n_read,n_unique,n_dup=ingest(gz,db,pool_filter)
    print(f"  read {n_read} unique {n_unique} duplicates {n_dup}",flush=True)
    con=sqlite3.connect(db); cur=con.cursor()
    toks=set()
    for (t,) in cur.execute("SELECT DISTINCT token0 FROM events"): toks.add(t)
    for (t,) in cur.execute("SELECT DISTINCT token1 FROM events"): toks.add(t)
    print(f"resolving decimals for {len(toks)} tokens...",flush=True)
    dec,dec_defaulted=load_decimals(toks)
    def scale(x,tok):
        try: return int(x)/(10**dec.get(tok,18))
        except: return 0.0
    pools=[r[0] for r in cur.execute("SELECT DISTINCT pool FROM events")]
    panel=[]
    for pi,p in enumerate(pools):
        book=Book()
        for t,v in tickbook.get(p,{}).items(): book.apply(int(t),int(v))
        cur_L=0; cur_tick=None; last_ts=None; role=role_of.get(p,"unknown")
        wk=defaultdict(lambda: defaultdict(float))
        liq=defaultdict(lambda: defaultdict(int))  # exact integer liquidity sums (mint/burn/jit); float loses precision >2^53
        open_pos={}; sameblk=defaultdict(dict); owners_wk=defaultdict(set)
        q=cur.execute("""SELECT ts,type,owner,tickLower,tickUpper,liquidity_delta,swap_liquidity,
                         amount0,amount1,tick,block,token0,token1 FROM events
                         WHERE pool=? ORDER BY block,tx_index,log_index""",(p,))
        while True:
            rows=q.fetchmany(20000)
            if not rows: break
            for (ts,typ,owner,tl,tu,ldelta,swl,a0,a1,tk,blk,t0,t1) in rows:
                if ts is None: continue
                if last_ts is not None and cur_L>0 and ts>last_ts:
                    a=last_ts
                    while a<ts:
                        wend=week_start(a)+SECWK; seg=min(ts,wend)-a
                        wk[week_id(a)]["twl_num"]+=cur_L*seg; wk[week_id(a)]["twl_den"]+=seg
                        a=min(ts,wend)
                last_ts=ts; W=week_id(ts)
                if typ=="swap":
                    v0=abs(scale(a0,t0)); v1=abs(scale(a1,t1))
                    wk[W]["swaps"]+=1; wk[W]["vol0"]+=v0; wk[W]["vol1"]+=v1
                    cur_L=int(swl) if swl not in ("",None) else cur_L
                    if tk is not None: cur_tick=tk
                    wk[W]["fee_income"]+=v1*tier.get(p,0)/1e6
                    if cur_tick is not None:
                        for b,wd in BANDS.items():
                            wk[W]["depth_"+b]=(book.active_L(cur_tick-wd)+book.active_L(cur_tick)+book.active_L(cur_tick+wd))/3.0
                elif typ in ("mint","burn"):
                    L=int(ldelta) if ldelta not in ("",None) else 0
                    sgn=1 if typ=="mint" else -1; key=(owner,tl,tu)
                    if tl is not None and tu is not None:
                        book.apply(tl,sgn*L); book.apply(tu,-sgn*L)
                    owners_wk[W].add(owner)
                    if typ=="mint":
                        wk[W]["lp_entry"]+=1; liq[W]["mint_liq"]+=L; open_pos[key]=ts
                        sameblk[blk][key]=sameblk[blk].get(key,0)+L
                    else:
                        wk[W]["lp_exit"]+=1; liq[W]["burn_liq"]+=L
                        if key in open_pos: wk[W]["dur_sum"]+=(ts-open_pos.pop(key)); wk[W]["dur_n"]+=1
                        if key in sameblk.get(blk,{}): liq[W]["jit_liq"]+=min(L,sameblk[blk][key])
                elif typ=="collect":
                    wk[W]["collects"]+=1; wk[W]["collect_amt1"]+=abs(scale(a1,t1))
        for W,a in wk.items():
            twl=a["twl_num"]/a["twl_den"] if a.get("twl_den",0)>0 else 0.0
            fee=a.get("fee_income",0)
            lq=liq.get(W,{}); mint_liq=lq.get("mint_liq",0); burn_liq=lq.get("burn_liq",0)
            panel.append({"pool":p,"unit_role":role,"week":W,
                "swaps":int(a.get("swaps",0)),"vol0":round(a.get("vol0",0),6),"vol1":round(a.get("vol1",0),6),
                "twl_active_liquidity":round(twl,2),
                "depth_1pct":round(a.get("depth_1pct",0),2),"depth_2pct":round(a.get("depth_2pct",0),2),
                "depth_5pct":round(a.get("depth_5pct",0),2),
                "lp_entry_count":int(a.get("lp_entry",0)),"lp_exit_count":int(a.get("lp_exit",0)),
                "unique_lp_count":len(owners_wk.get(W,set())),
                "jit_share_same_block":round(lq.get("jit_liq",0)/mint_liq,4) if mint_liq>0 else 0.0,
                "lp_fee_income_native1":round(fee,6),
                "lp_fee_income_per_active_liquidity":round(fee/twl,12) if twl>0 else 0.0,
                "collect_amount1_native":round(a.get("collect_amt1",0),6),
                "position_duration_days":round(a.get("dur_sum",0)/a["dur_n"]/86400,2) if a.get("dur_n",0)>0 else 0.0,
                "net_liq":mint_liq-burn_liq})
        if pi%100==0: print(f"  built {pi}/{len(pools)} pools, {len(panel)} rows",flush=True)
    con.close()
    return panel, {"n_events_read":n_read,"n_events_unique":n_unique,"n_duplicates":n_dup,
                   "decimals_defaulted_tokens":len(dec_defaulted),"decimals_defaulted_list":dec_defaulted[:50]}

def write_artifacts(panel, ingest_meta, prefix="", unit_set=UNIT_SET):
    cols=list(panel[0].keys())
    outcols=[c for c in cols if c not in ("pool","unit_role","week")]
    csvp=os.path.join(DATA,f"panel_weekly_{prefix}frozen.csv")
    with open(csvp,"w",newline="") as g:
        w=csv.DictWriter(g,fieldnames=cols); w.writeheader(); w.writerows(panel)
    try:
        import polars as pl
        pl.DataFrame(panel).write_parquet(os.path.join(DATA,f"panel_weekly_{prefix}frozen.parquet"))
    except Exception as e:
        print(f"  parquet skipped: {e}",flush=True)

    # W-3 completeness: expected (unit set x week grid) vs observed, by role & outcome
    observed=defaultdict(set)                       # pool -> set(weeks) present
    for r in panel: observed[r["pool"]].add(r["week"])
    roles=sorted({role_of.get(p,"unknown") for p in unit_set})
    exp_by_role=Counter(); obs_by_role=Counter()
    for p in unit_set:
        role=role_of.get(p,"unknown")
        exp_by_role[role]+=len(WEEKS)
        obs_by_role[role]+=len(observed.get(p,set()))
    pools_with_data={r["pool"] for r in panel}
    units_no_data=sorted(unit_set-pools_with_data)
    # outcome-specific missingness: among observed rows, count zero/absent per role x outcome
    role_ct=Counter(r["unit_role"] for r in panel)
    outcome_missing={}
    for c in outcols:
        outcome_missing[c]={rl:sum(1 for r in panel if r["unit_role"]==rl and (r[c]==0 or r[c]==0.0)) for rl in role_ct}
    # failed RPC ranges (events pull + tickbook init)
    ck=json.load(open(os.path.join(DATA,"ckpt_events.json"))) if os.path.exists(os.path.join(DATA,"ckpt_events.json")) else {}
    sir=json.load(open(os.path.join(DATA,"state_initialization_report.json"))) if os.path.exists(os.path.join(DATA,"state_initialization_report.json")) else {}
    init_missing_units=sorted(unit_set-set(tickbook.keys()))
    comp={
      "frozen_unit_set_size":len(unit_set),
      "frozen_week_grid_size":len(WEEKS),
      "expected_pool_weeks_total":len(unit_set)*len(WEEKS),
      "observed_pool_weeks_total":len(panel),
      "expected_pool_weeks_by_role":dict(exp_by_role),
      "observed_pool_weeks_by_role":dict(obs_by_role),
      "missing_pool_weeks_by_role":{r:exp_by_role[r]-obs_by_role[r] for r in roles},
      "units_with_zero_observed_weeks":{"count":len(units_no_data),"sample":units_no_data[:30]},
      "outcome_specific_zero_counts_by_role":outcome_missing,
      "events_read":ingest_meta["n_events_read"],
      "events_unique":ingest_meta["n_events_unique"],
      "duplicate_events_dropped":ingest_meta["n_duplicates"],
      "decimals_defaulted_to_18":ingest_meta["decimals_defaulted_tokens"],
      "events_pull_failed_ranges":ck.get("failed",[]) if isinstance(ck.get("failed"),list) else ck.get("failed",0),
      "tickbook_incomplete_init_pools":sir.get("n_incomplete",None),
      "tickbook_init_missing_units":{"count":len(init_missing_units),"sample":init_missing_units[:30]},
    }
    json.dump(comp,open(os.path.join(DATA,f"panel_{prefix}completeness.json"),"w"),indent=1)

    # sanity
    import statistics as st
    def col(c): return [r[c] for r in panel if isinstance(r[c],(int,float))]
    sane={c:{"min":min(col(c)),"max":max(col(c)),"median":st.median(col(c)),
             "nonzero_frac":round(sum(1 for x in col(c) if x)/len(panel),4)} for c in outcols}
    sane["_units_note"]="lp_fee_income_native1 and _per_active_liquidity are in native token1 units (fee_tier x |amount1_human|); USD pricing deferred, do not treat as USD."
    json.dump(sane,open(os.path.join(DATA,f"outcome_{prefix}sanity_report.json"),"w"),indent=1)

    man={}
    for f in [f"panel_weekly_{prefix}frozen.csv",f"panel_weekly_{prefix}frozen.parquet",
              f"panel_{prefix}completeness.json",f"outcome_{prefix}sanity_report.json",
              "state_initialization_report.json","tickbook_init.json","token_decimals.json"]:
        fp=os.path.join(DATA,f)
        if os.path.exists(fp):
            man[f]={"sha256":hashlib.sha256(open(fp,"rb").read()).hexdigest(),"bytes":os.path.getsize(fp)}
    json.dump(man,open(os.path.join(DATA,f"artifact_{prefix}manifest.json"),"w"),indent=1)
    print(f"OUTCOME PANEL {len(panel)} pool-weeks; roles {dict(role_ct)}",flush=True)
    return comp, sane

if __name__=="__main__":
    panel,meta=build()
    if not panel:
        print("NO PANEL ROWS — aborting artifact write",flush=True); raise SystemExit(1)
    write_artifacts(panel,meta)
    print("BUILD DONE",flush=True)
