"""migratory-birds level 3. Flock -> species. Each file has one flock per species (rectangular assignment for the 2-flock example).
Species rules from the ornithologists' calls (scores in 0..1), then scipy linear_sum_assignment:
 Medieval Bluetit    : every row identical and a palindrome (back along the same route), not contained in another flock's BOPs
 Sticky Wolfthroat   : BOPs all lie on another flock's route (segments of the Bluetit route)
 Flanking Blackfinch : identical rows, closed non-palindromic loop (orbit, stay together), not lying on another flock's route
 Rusty Goldhammer    : long shared prefix (journey together), rows then differ (split up)
 Red Firefinch       : closed loops from one nest, rows differ, middles do not overlap (own adventures)
 Hurracurra Bird     : closed loops, rows differ, middles drawn from the same pool of BOPs, hot BOPs (temperature after the degree F fix)"""
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from cf import submit
from . import l1

SPECIES = ["Hurracurra Bird", "Medieval Bluetit", "Flanking Blackfinch", "Rusty Goldhammer", "Red Firefinch", "Sticky Wolfthroat"]
HEADER = "Flock ID,Species"

def read_flocks(path):
    df = pd.read_csv(path, dtype=str, skipinitialspace=True); df.columns = ["fid", "path"]
    df["p"] = df.path.map(lambda s: [int(x) for x in s.split()]); df["fid"] = df.fid.astype(int); return df

def temp_map(ctx):
    T = l1.load(Path(ctx.level.dir) / "all_data_from_level_1.in"); T["tC"] = l1.to_celsius(T.temp)
    return dict(zip(T.bop, T.tC)), float(T.tC.mean())

def features(df, tmap):
    sets = {fid: {x for p in g.p for x in p} for fid, g in df.groupby("fid")}; rows = []
    for fid, g in df.groupby("fid"):
        P = list(g.p); n = len(P); L = float(np.mean([len(p) for p in P]))
        ident = all(p == P[0] for p in P); pal = all(p == p[::-1] for p in P); loop = all(p[0] == p[-1] for p in P)
        pre = 0
        while pre < min(map(len, P)) and all(p[pre] == P[0][pre] for p in P): pre += 1
        pre = 0 if ident else pre
        tails = [set(p[pre:-1]) - {p[0]} for p in P]
        J = float(np.mean([len(a & b) / max(1, len(a | b)) for a, b in combinations(tails, 2)])) if n > 1 else 0.0
        cont_by = max((len(sets[fid] & sets[o]) / len(sets[fid]) for o in sets if o != fid), default=0.0)
        rows.append(dict(fid=fid, rows=n, len=L, ident=ident, pal=pal, loop=loop, pre_x=max(0, pre - 1) / L, J=J, cont_by=cont_by,
                         tC=float(np.mean([tmap[b] for b in sets[fid] if b in tmap]))))
    return pd.DataFrame(rows)

def score_matrix(f, gmean):
    S = pd.DataFrame(0.0, index=f.fid, columns=SPECIES)
    for r in f.itertuples():
        i = r.fid; ident, pal, loop = float(r.ident), float(r.pal), float(r.loop)
        pre = min(1.0, r.pre_x / 0.4); hot = float(np.clip((r.tC - gmean) / (40 - gmean), 0, 1)); J = min(1.0, r.J / 0.15)
        S.loc[i, "Medieval Bluetit"] = ident * pal * (1 - r.cont_by)
        S.loc[i, "Sticky Wolfthroat"] = r.cont_by ** 2
        S.loc[i, "Flanking Blackfinch"] = ident * (1 - pal) * loop * (1 - r.cont_by)  # orbits its own land, not a sub-route of another flock
        S.loc[i, "Rusty Goldhammer"] = (1 - ident) * pre
        S.loc[i, "Red Firefinch"] = (1 - ident) * loop * (1 - J) * (1 - pre)
        S.loc[i, "Hurracurra Bird"] = (1 - ident) * (0.5 * J + 0.5 * hot)
    return S

def assign(path, ctx):
    tmap, gmean = temp_map(ctx); f = features(read_flocks(path), tmap); S = score_matrix(f, gmean)
    r, c = linear_sum_assignment(-S.values)
    return {int(S.index[i]): SPECIES[j] for i, j in zip(r, c)}, S, f

def solve(path, ctx) -> str:
    a, _, _ = assign(path, ctx)
    return "\n".join([HEADER] + [f"{k},{a[k]}" for k in sorted(a)])
