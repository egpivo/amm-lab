import json, os, csv, math, time, urllib.request
from collections import defaultdict, Counter
ROOT="/Users/joseph/amm-lab"
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
rows=list(csv.DictReader(open("feerev_panelvars.csv")))
# backfill tier for any row with tier 0 (untreated); cache
tier_cache=json.load(open("ckpt_tier_bf.json")) if os.path.exists("ckpt_tier_bf.json") else {}
need=[r for r in rows if r["tier"] in ("0","") ]
print("rows needing tier:", len(need), flush=True)
for i,r in enumerate(need):
    p=r["pool"]
    if p in tier_cache: r["tier"]=str(tier_cache[p]); continue
    v=rpc("eth_call",[{"to":p,"data":"0xddca3f43"},"latest"])
    t=int(v,16) if v and v!="0x" else 0
    tier_cache[p]=t; r["tier"]=str(t)
    if i%150==0: json.dump(tier_cache,open("ckpt_tier_bf.json","w")); print("  tier",i,flush=True)
json.dump(tier_cache,open("ckpt_tier_bf.json","w"))
w=csv.DictWriter(open("feerev_panelvars.csv","w",newline=""),fieldnames=list(rows[0].keys()))
w.writeheader(); w.writerows(rows)
print("tier backfill done", flush=True)

# rematch with spec exposure + correct tiers
tok=json.load(open("ckpt_tokens.json"))
treated={r["pool"] for r in rows if r["treated"]=="1"}
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
cands=defaultdict(list)
for r in rows:
    if r["treated"]=="0" and r["old"]=="1" and r["covered"]=="1" and float(r["fr12_usd"])>0 and exp(r["pool"])<=0.25:
        cands[(r["tier"],r["class"])].append(r)
matched=[]; unmatched=[]
for r in rows:
    if not (r["treated"]=="1" and r["old"]=="1" and r["covered"]=="1" and float(r["fr12_usd"])>0): continue
    pool=cands.get((r["tier"],r["class"]),[]); lg=math.log(float(r["fr12_usd"]))
    near=sorted(pool,key=lambda c:abs(math.log(float(c["fr12_usd"]))-lg))[:3]
    near=[c for c in near if abs(math.log(float(c["fr12_usd"]))-lg)<=0.5]
    if near: matched.append({"treated":r["pool"],"fr12":float(r["fr12_usd"]),"tier":r["tier"],"class":r["class"],"controls":[c["pool"] for c in near]})
    else: unmatched.append({"pool":r["pool"],"fr12":float(r["fr12_usd"]),"tier":r["tier"],"class":r["class"]})
n_main=sum(1 for r in rows if r["treated"]=="1" and r["old"]=="1" and r["covered"]=="1" and float(r["fr12_usd"])>0)
byc=Counter((m["tier"],m["class"]) for m in matched)
# caliper sensitivity
def rate(cal):
    m=0
    for r in rows:
        if not (r["treated"]=="1" and r["old"]=="1" and r["covered"]=="1" and float(r["fr12_usd"])>0): continue
        pool=cands.get((r["tier"],r["class"]),[]); lg=math.log(float(r["fr12_usd"]))
        if any(abs(math.log(float(c["fr12_usd"]))-lg)<=cal for c in pool): m+=1
    return m,n_main
rep={"n_treated_main_sample":n_main,"n_matched_treated":len(matched),
     "match_rate_main":round(len(matched)/max(1,n_main),3),
     "n_candidate_pure_controls":sum(len(v) for v in cands.values()),
     "n_pure_controls_used":len({c for m in matched for c in m["controls"]}),
     "match_by_tier_class":{f"{k[0]}|{k[1]}":v for k,v in sorted(byc.items())},
     "caliper_sensitivity":{str(cal):rate(cal)[0] for cal in [0.5,1.0,1.5,2.0]},
     "unmatched_top":[{"pool":u["pool"][:10],"fr12":int(u["fr12"]),"tier":u["tier"],"class":u["class"]} for u in sorted(unmatched,key=lambda x:-x["fr12"])[:10]]}
json.dump(rep,open("match_feasibility.json","w"),indent=1)
json.dump(matched,open("matched_pairs.json","w"),indent=1)
print(json.dumps(rep,indent=1),flush=True)
