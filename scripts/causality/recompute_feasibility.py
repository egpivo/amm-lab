"""B2 fix: enforce canonical-factory controls, recompute matching/feasibility/panel_units.

Supersedes backfill_tier_rematch matching. Reads feerev_panelvars (tier already
backfilled), tokens, prices; backfills factory() for reservoir pools; filters primary
controls to the canonical Uniswap v3 factory; re-runs exact-tier+pair-class matching,
NN on log pre-period fee revenue (caliper 0.5); recomputes SMD, feasibility report,
matched_pairs, panel_units. Design-only, pre-period; no post-period data.
"""
import json, os, csv, math, time, urllib.request, hashlib, statistics as st
from collections import defaultdict, Counter
ROOT="/Users/joseph/amm-lab"; DATA=os.path.join(ROOT,".local/amm_paper_c/data")
CANON="0x1f98431c8ad98523631ae4a59f267346ea31f984"
def rpcurl():
    for l in open(os.path.join(ROOT,".env")):
        if l.startswith("ALCHEMY_ETHEREUM_URL"): return l.split("=",1)[1].strip().strip('"')
RPC=rpcurl()
def rpc(m,p,retries=4):
    d=json.dumps({"jsonrpc":"2.0","id":1,"method":m,"params":p}).encode()
    for a in range(retries):
        try:
            r=urllib.request.Request(RPC,data=d,headers={"Content-Type":"application/json"})
            o=json.loads(urllib.request.urlopen(r,timeout=25).read())
            if "error" in o: raise RuntimeError
            return o["result"]
        except Exception:
            if a==retries-1: return None
            time.sleep(1)
rows={r["pool"]:r for r in csv.DictReader(open(os.path.join(DATA,"feerev_panelvars.csv")))}
tok=json.load(open(os.path.join(DATA,"ckpt_tokens.json")))
treated={p for p,r in rows.items() if r["treated"]=="1"}

# reservoir = untreated, old, covered, fr12>0
reservoir=[p for p,r in rows.items() if r["treated"]=="0" and r["old"]=="1" and r["covered"]=="1" and float(r["fr12_usd"])>0]
# exposure (spec): same pair -> spillover(1.0); shared token -> 0.125; else 0
tpairs=set(); ttok=set()
for p in treated:
    t=tok.get(p,[None,None])
    if t[0]: tpairs.add(frozenset(t)); ttok.update(t)
def exp(p):
    t=tok.get(p,[None,None])
    if not t[0]: return 1.0
    if frozenset(t) in tpairs: return 1.0
    if t[0] in ttok or t[1] in ttok: return 0.125
    return 0.0

# B2: backfill factory() for reservoir, keep canonical only
fac=json.load(open(os.path.join(DATA,"ckpt_factory.json"))) if os.path.exists(os.path.join(DATA,"ckpt_factory.json")) else {}
todo=[p for p in reservoir if p not in fac]
print("factory backfill needed:",len(todo),flush=True)
for i,p in enumerate(todo):
    v=rpc("eth_call",[{"to":p,"data":"0xc45a0155"},"latest"])   # factory()
    fac[p]="0x"+v[-40:] if v and v!="0x" else None
    if i%150==0: json.dump(fac,open(os.path.join(DATA,"ckpt_factory.json"),"w")); print("  fac",i,flush=True)
json.dump(fac,open(os.path.join(DATA,"ckpt_factory.json"),"w"))

canon_reservoir=[p for p in reservoir if fac.get(p)==CANON and exp(p)<=0.25]
noncanon=[p for p in reservoir if fac.get(p)!=CANON]
print(f"reservoir {len(reservoir)} -> canonical+pure {len(canon_reservoir)}; forks (to cross-venue) {len(noncanon)}",flush=True)

cands=defaultdict(list)
for p in canon_reservoir:
    r=rows[p]; cands[(r["tier"],r["class"])].append(p)
def lfr(p): return math.log(float(rows[p]["fr12_usd"]))
treated_main=[p for p,r in rows.items() if r["treated"]=="1" and r["old"]=="1" and r["covered"]=="1" and float(r["fr12_usd"])>0]
matched=[]; unmatched=[]
for p in treated_main:
    r=rows[p]; pool=cands.get((r["tier"],r["class"]),[]); lg=lfr(p)
    near=sorted(pool,key=lambda c:abs(lfr(c)-lg))[:3]
    near=[c for c in near if abs(lfr(c)-lg)<=0.5]
    if near: matched.append({"treated":p,"fr12":float(r["fr12_usd"]),"tier":r["tier"],"class":r["class"],"controls":near})
    else: unmatched.append({"pool":p,"fr12":float(r["fr12_usd"]),"tier":r["tier"],"class":r["class"]})
unmatched.sort(key=lambda x:-x["fr12"])
ctrl_used=sorted({c for m in matched for c in m["controls"]})

def smd(A,B,f):
    xa=[f(x) for x in A]; xb=[f(x) for x in B]
    sp=math.sqrt((st.pstdev(xa)**2+st.pstdev(xb)**2)/2) or 1e-9
    return (st.mean(xa)-st.mean(xb))/sp
tm=[m["treated"] for m in matched]
def lsw(p): return math.log(float(rows[p]["sw12"])) if float(rows[p]["sw12"])>0 else 0.0
def cal(c):
    n=0
    for p in treated_main:
        r=rows[p]; pool=cands.get((r["tier"],r["class"]),[]); lg=lfr(p)
        if any(abs(lfr(x)-lg)<=c for x in pool): n+=1
    return n
byc=Counter((m["tier"],m["class"]) for m in matched)
rep={"canonical_only":True,"n_treated_main":len(treated_main),"n_matched":len(matched),
     "match_rate":round(len(matched)/len(treated_main),3),
     "n_candidate_pure_controls":len(canon_reservoir),"n_controls_used":len(ctrl_used),
     "n_unmatched":len(unmatched),
     "smd_logfr_before":round(smd(tm,canon_reservoir,lfr),3),
     "smd_logfr_after":round(smd(tm,ctrl_used,lfr),3) if ctrl_used else None,
     "smd_logsw_before":round(smd(tm,canon_reservoir,lsw),3),
     "smd_logsw_after":round(smd(tm,ctrl_used,lsw),3) if ctrl_used else None,
     "caliper_sensitivity":{str(c):cal(c) for c in [0.5,1.0,1.5,2.0]},
     "match_by_tier_class":{f"{k[0]}|{k[1]}":v for k,v in sorted(byc.items())},
     "top_unmatched":[{"pool":u["pool"][:10],"fr12":int(u["fr12"]),"tier":u["tier"],"class":u["class"]} for u in unmatched[:15]]}
json.dump(rep,open(os.path.join(DATA,"match_feasibility.json"),"w"),indent=1)
json.dump(matched,open(os.path.join(DATA,"matched_pairs.json"),"w"),indent=1)
units={"treated_matched":sorted(tm),"controls":ctrl_used,
       "unmatched_treated":sorted(u["pool"] for u in unmatched),
       "treated_main":sorted(treated_main),
       "crossvenue_forks":sorted(noncanon)}
json.dump(units,open(os.path.join(DATA,"panel_units.json"),"w"),indent=1)
print(json.dumps(rep,indent=1),flush=True)
print("RECOMPUTE DONE",flush=True)
