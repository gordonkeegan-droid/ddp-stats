"""Build players.json for the DDP HR app.
Sandbox mode: 2021-2022 GitHub mirror (demo vintage).
Local mode:   pass --local to pull current seasons via pybaseball.

Output: league pools + per-hitter empirical (EV,LA,spray) samples by pitcher
hand + PA/BBE rates + per-pitcher contact-quality logit shifts (shrunk).
"""
import pandas as pd, numpy as np, json, sys, urllib.request, os

LOCAL = "--local" in sys.argv
KEEP = ["game_date","game_type","batter","pitcher","stand","p_throws","events",
        "type","launch_speed","launch_angle","hc_x","hc_y"]
PA_EVENTS = set("""strikeout field_out single double triple home_run walk hit_by_pitch
force_out grounded_into_double_play double_play sac_fly sac_bunt field_error
fielders_choice fielders_choice_out strikeout_double_play sac_fly_double_play
triple_play catcher_interf other_out""".split())

def load():
    if LOCAL:
        from pybaseball import statcast, cache; cache.enable()
        frames=[statcast(start_dt=a,end_dt=b)[KEEP] for a,b in
                [("2024-03-20","2024-09-29"),("2025-03-27","2025-09-28"),
                 ("2026-03-26","2026-08-03")]]
        return pd.concat(frames, ignore_index=True)
    BASE="https://raw.githubusercontent.com/sportsdataverse/baseballr-data/main/statcast"
    frames=[]
    for yr,ms in {2021:["April","May","June","July","August","September","October"],
                  2022:["April","May","June","July","August"]}.items():
        for m in ms:
            fn=f"{yr}-{m}-StatcastData.parquet"; tmp=f"/tmp/{fn}"
            urllib.request.urlretrieve(f"{BASE}/{fn}",tmp)
            d=pd.read_parquet(tmp,columns=KEEP); os.remove(tmp)
            frames.append(d)
            print(fn, len(d))
    return pd.concat(frames, ignore_index=True)

df = load()
df = df[df["game_type"]=="R"].copy()
df["game_date"] = pd.to_datetime(df["game_date"])

# ---------- PA and BBE counts ----------
pa = df[df["events"].isin(PA_EVENTS)].copy()
bb = df[(df["type"]=="X")].dropna(subset=["launch_speed","launch_angle","hc_x","hc_y"]).copy()
phi = np.degrees(np.arctan((bb["hc_x"]-125.42)/(198.27-bb["hc_y"])))
bb["spray_b"] = np.where(bb["stand"]=="R",-phi,phi)   # batter-relative, +=pull
bb = bb[bb["launch_speed"].between(30,125)&bb["launch_angle"].between(-90,90)&bb["spray_b"].between(-60,60)]
# drop Savant-imputed clusters
pc = bb.groupby(["launch_speed","launch_angle"]).size()
bb = bb[~bb.set_index(["launch_speed","launch_angle"]).index.isin(set(pc[pc>400].index))]

# recency weight (half-life ~ 1 season)
age_days = (bb["game_date"].max()-bb["game_date"]).dt.days
bb["w"] = np.exp(-age_days/270.0)

# ---------- league fence-logit machinery (for pitcher shifts) ----------
fm = json.load(open("output/fence_model.json"))
g_ev=np.array(fm["grid"]["ev"]); g_la=np.array(fm["grid"]["la"]); g_sp=np.array(fm["grid"]["spray"])
G = np.array(fm["grid"]["p"])
def interp_logit(ev,la,sp):
    p=[]
    ev=np.clip(ev,g_ev[0],g_ev[-1]); la=np.clip(la,g_la[0],g_la[-1]); sp=np.clip(sp,g_sp[0],g_sp[-1])
    ie=np.clip(np.searchsorted(g_ev,ev)-1,0,len(g_ev)-2)
    il=np.clip(np.searchsorted(g_la,la)-1,0,len(g_la)-2)
    isp=np.clip(np.searchsorted(g_sp,sp)-1,0,len(g_sp)-2)
    fe=(ev-g_ev[ie])/(g_ev[ie+1]-g_ev[ie]); fl=(la-g_la[il])/(g_la[il+1]-g_la[il]); fs=(sp-g_sp[isp])/(g_sp[isp+1]-g_sp[isp])
    v=0
    for de,we in ((0,1-fe),(1,fe)):
        for dl,wl in ((0,1-fl),(1,fl)):
            for ds,ws in ((0,1-fs),(1,fs)):
                v=v+we*wl*ws*G[ie+de,il+dl,isp+ds]
    v=np.clip(v,1e-4,1-1e-4)
    return np.log(v/(1-v))

bb["spray_f"] = np.where(bb["stand"]=="R",-bb["spray_b"],bb["spray_b"])
bb["z"] = interp_logit(bb["launch_speed"].values, bb["launch_angle"].values, bb["spray_f"].values)
LEAGUE_Z = float(np.average(bb["z"], weights=bb["w"]))
LEAGUE_BBE_PA = len(bb)/len(pa)

# ---------- hitters ----------
def pack(sdf, nmax=120):
    if len(sdf)==0: return None
    take=min(nmax,len(sdf))
    idx=np.random.default_rng(7).choice(sdf.index, size=take, replace=False,
        p=(sdf["w"]/sdf["w"].sum()).values)
    s=sdf.loc[idx]
    return {"n": int(len(sdf)),
            "stand": s["stand"].mode()[0],
            "s": np.round(s[["launch_speed","launch_angle","spray_b"]].values,1).tolist()}

hitters={}
pa_h = pa.groupby(["batter","p_throws"]).size()
k_pa=60
for bid, g in bb.groupby("batter"):
    if len(g)<80: continue
    h={}
    for hand in ("L","R"):
        gg=g[g["p_throws"]==hand]
        blk=pack(gg)
        if blk is None: continue
        npa=int(pa_h.get((bid,hand),0))
        blk["bbe_pa"]=round((len(gg)+LEAGUE_BBE_PA*k_pa)/(max(npa,len(gg))+k_pa),3)
        h["vs"+hand]=blk
    if h: hitters[str(int(bid))]=h

# ---------- pitchers ----------
pitchers={}
pa_p = pa.groupby("pitcher").size(); k_z=400; k_pa_p=100
for pid, g in bb.groupby("pitcher"):
    if len(g)<60: continue
    zbar=float(np.average(g["z"],weights=g["w"]))
    shift=(zbar-LEAGUE_Z)*len(g)/(len(g)+k_z)
    npa=int(pa_p.get(pid,0))
    pitchers[str(int(pid))]={"throws":g["p_throws"].mode()[0],
        "shift":round(shift,4), "n":int(len(g)),
        "bbe_pa":round((len(g)+LEAGUE_BBE_PA*k_pa_p)/(max(npa,len(g))+k_pa_p),3)}

# ---------- league pools by batter side ----------
pools={}
for st in ("L","R"):
    s=bb[bb["stand"]==st].sample(450, random_state=7)
    pools[st]=np.round(s[["launch_speed","launch_angle","spray_b"]].values,1).tolist()

out={"meta":{"vintage":"2021-2022 DEMO -- rebuild with --local for current ball",
             "league_bbe_pa":round(LEAGUE_BBE_PA,3),"league_z":round(LEAGUE_Z,3),
             "shrink_k_bbe":80},
     "pools":pools,"hitters":hitters,"pitchers":pitchers}
json.dump(out,open("output/players.json","w"))
print(f"\nhitters {len(hitters)}  pitchers {len(pitchers)}  league bbe/pa {LEAGUE_BBE_PA:.3f}")
print("size:", round(os.path.getsize("output/players.json")/1e6,2),"MB")
