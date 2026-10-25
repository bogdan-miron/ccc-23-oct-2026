"""migratory-birds level 4. Classify flocks (not rows) from their paths. Features per flock encode the L3 species rules in a noise-tolerant way;
a classifier is trained on all labeled flocks. Output: one row per flock with missing species."""
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from cf import io as cio
from . import l1

SPECIES = ["Hurracurra Bird", "Medieval Bluetit", "Flanking Blackfinch", "Rusty Goldhammer", "Red Firefinch", "Sticky Wolfthroat"]
HEADER = "Flock ID,Species"

def read_rows(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    df.columns = ["fid", "path", "sp"][:df.shape[1]]
    df["fid"] = df.fid.astype(int); df["p"] = df.path.map(lambda s: [int(x) for x in s.split()])
    if "sp" not in df: df["sp"] = "missing"
    return df

def temp_map(level_dir):
    T = l1.load(Path(level_dir) / "all_data_from_level_1.in"); T["tC"] = l1.to_celsius(T.temp)
    return dict(zip(T.bop, T.tC)), float(T.tC.mean())

def _cp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

def flock_features(df, tmap, ref_sets=None):
    """df: rows of all flocks in the file. Returns DataFrame indexed by fid. Cross-flock features use the other flocks of the same file."""
    groups = {fid: list(g.p) for fid, g in df.groupby("fid")}
    sets = {fid: {x for p in P for x in p} for fid, P in groups.items()}
    pal_frac = {fid: np.mean([p == p[::-1] for p in P]) for fid, P in groups.items()}
    start_count = pd.Series({fid: P[0][0] for fid, P in groups.items()}).value_counts()
    bop_flocks = {}
    for fid, s in sets.items():
        for b in s: bop_flocks.setdefault(b, set()).add(fid)
    edges = {fid: {frozenset(e) for p in P for e in zip(p, p[1:])} for fid, P in groups.items()}
    rows = []
    for fid, P in groups.items():
        n = len(P); L = np.array([len(p) for p in P], float); S = sets[fid]
        pairs = list(combinations(range(n), 2))
        cp = [_cp(P[i], P[j]) for i, j in pairs]; cs = [_cp(P[i][::-1], P[j][::-1]) for i, j in pairs]
        mL = [min(len(P[i]), len(P[j])) for i, j in pairs]
        sm = [SequenceMatcher(None, P[i], P[j], autojunk=False).ratio() for i, j in pairs]
        jac = [len(set(P[i]) & set(P[j])) / len(set(P[i]) | set(P[j])) for i, j in pairs]
        tail = [len((set(P[i][c:-1]) - {P[i][0]}) & (set(P[j][c:-1]) - {P[j][0]})) / max(1, len((set(P[i][c:-1]) - {P[i][0]}) | (set(P[j][c:-1]) - {P[j][0]}))) for (i, j), c in zip(pairs, cp)]
        modal = max(set(map(tuple, P)), key=lambda t: sum(tuple(p) == t for p in P))
        half = [np.mean([p[k] == p[-1 - k] for k in range(len(p) // 2)]) for p in P]
        others = [o for o in sets if o != fid]
        cont = np.array([len(S & sets[o]) / len(S) for o in others]); cont_rev = np.array([len(S & sets[o]) / len(sets[o]) for o in others])
        pal_union = set().union(*[sets[o] for o in others if pal_frac[o] > 0.8]) if any(pal_frac[o] > 0.8 for o in others) else set()
        pal_edges = set().union(*[edges[o] for o in others if pal_frac[o] > 0.8]) if any(pal_frac[o] > 0.8 for o in others) else set()
        oth_edges = set().union(*[edges[o] for o in others if pal_frac[o] <= 0.8]) if any(pal_frac[o] <= 0.8 for o in others) else set()
        e_pal = np.mean([e in pal_edges for e in edges[fid]]); e_oth = np.mean([e in oth_edges for e in edges[fid]])
        shared = np.mean([len(bop_flocks[b]) > 1 for b in S])
        tc = np.array([tmap[b] for b in S if b in tmap])
        rows.append(dict(fid=fid, n=n, len_mean=L.mean(), len_min=L.min(), len_max=L.max(), len_std=L.std(),
            pal_frac=pal_frac[fid], half_pal=np.mean(half), ident=np.mean([tuple(p) == modal for p in P]),
            cp_share=np.mean(np.array(cp) / np.array(mL)) if pairs else 0, cs_share=np.mean(np.array(cs) / np.array(mL)) if pairs else 0,
            seq_ratio=np.mean(sm) if pairs else 1, jac=np.mean(jac) if pairs else 1, tail_j=np.mean(tail) if pairs else 0,
            distinct=np.mean([len(set(p)) / len(p) for p in P]), n_set=len(S) / L.mean(),
            cont_by=cont.max() if len(cont) else 0, n_cont90=int((cont >= 0.9).sum()), cont_rev=cont_rev.max() if len(cont) else 0,
            in_pal_union=np.mean([b in pal_union for b in S]), edge_pal=e_pal, edge_oth=e_oth, shared_bop=shared, start_shared=int(start_count[P[0][0]]),
            tC_mean=tc.mean(), tC_med=np.median(tc), tC_hot=np.mean(tc > 35), tC_min=tc.min()))
    return pd.DataFrame(rows).set_index("fid")

from sklearn.ensemble import ExtraTreesClassifier
CORE = ["len_mean", "pal_frac", "half_pal", "ident", "cp_share", "cs_share", "seq_ratio", "jac", "tail_j", "distinct", "tC_mean", "tC_med", "tC_hot", "tC_min"]
EDGE = ["edge_pal", "edge_oth"]  # CV F1 macro 0.9966 on 288 labeled flocks; adding row-count / cross-flock overlap features lowered it to 0.988-0.992
_cache = {}

def _main_files(ctx):
    return [f for f in ctx.level.inputs if "example" not in f.name.lower()]

def _train(ctx):
    key = str(ctx.level.dir)
    if key in _cache: return _cache[key]
    tmap, _ = temp_map(ctx.level.dir)
    rows = pd.concat([read_rows(f) for f in _main_files(ctx)]); X = flock_features(rows, tmap)
    lab = rows.groupby("fid").sp.first(); keep = lab != "missing"
    cols = CORE + EDGE
    m = ExtraTreesClassifier(1000, random_state=0, n_jobs=2).fit(X.loc[keep, cols], lab[keep])
    _cache[key] = (m, cols, tmap, rows); return _cache[key]

def predict(path, ctx):
    """Returns DataFrame indexed by fid with columns species, conf for flocks whose species is missing in `path`."""
    m, cols, tmap, main = _train(ctx); r = read_rows(path)
    if Path(path) in _main_files(ctx): X = flock_features(main, tmap).loc[sorted(r.fid.unique())]
    else:  # example file: compute cross-flock features inside the full universe, flock ids shifted to avoid collisions
        r2 = r.assign(fid=r.fid + 100000); X = flock_features(pd.concat([main, r2]), tmap).loc[sorted(r2.fid.unique())]; X.index = X.index - 100000
    miss = r[r.sp == "missing"].fid.unique(); X = X.loc[sorted(miss)]
    pr = m.predict_proba(X[cols]); return pd.DataFrame({"species": m.classes_[pr.argmax(1)], "conf": pr.max(1)}, index=X.index)

def solve(path, ctx) -> str:
    out = predict(path, ctx)
    return "\n".join([HEADER] + [f"{fid},{sp}" for fid, sp in out.species.items()])
