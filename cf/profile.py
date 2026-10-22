"""Auto-EDA in ONE call. profile(path_or_df, target=None) prints a compact markdown report; compare(a, b) checks train/test (or regime) shift.
Reads raw strings first so dirty tokens (words-as-numbers, 'missing', stray text) are visible BEFORE cleaning."""
import re
import numpy as np, pandas as pd
from pathlib import Path
from cf import io as cio

SPARK = "▁▂▃▄▅▆▇█"
TIME_HINT = re.compile(r"^(day|date|time|t|step|week|month|year|hour|timestamp|epoch)\b", re.I)
ID_HINT = re.compile(r"(^|_|\b)(id|bop|station|user|flock|item|node|sensor|site)(\b|_|$)", re.I)

def _spark(vals):
    vals = np.asarray(vals, float); m = vals.max()
    return "".join(SPARK[int(v / m * 7)] if m else SPARK[0] for v in vals)

def _load_raw(src, nrows=None):
    if isinstance(src, pd.DataFrame): return src.astype(object).where(src.notna(), None), None
    df = pd.read_csv(src, dtype=str, keep_default_na=False, na_values=[], skipinitialspace=True, nrows=nrows)
    return df, Path(src).stat().st_size

def _col_type(s):
    nn = s[~s.isin(cio.MISSING)].dropna()
    if len(nn) == 0: return "empty"
    num = pd.to_numeric(nn, errors="coerce").notna().mean()
    if num > 0.9: return "numeric"
    spl = nn.head(2000).map(lambda x: len(str(x).split()) > 1 and all(re.fullmatch(r"-?\d+(\.\d+)?", t) for t in str(x).split())).mean()
    if spl > 0.9: return "list"
    words = nn.head(2000).map(lambda x: cio.word2num(x) is not None).mean()
    if num + words > 0.9: return "numeric"
    return "categorical"

def profile(src, target=None, sample=200_000, top_corr=8, out=None):
    raw, size = _load_raw(src); n = len(raw); L = []
    name = Path(src).name if not isinstance(src, pd.DataFrame) else "DataFrame"
    L.append(f"## {name}: {n:,} rows x {raw.shape[1]} cols" + (f", {size/1e6:.1f} MB" if size else ""))
    types = {c: _col_type(raw[c]) for c in raw.columns}
    miss = {c: int(raw[c].isin(cio.MISSING).sum()) for c in raw.columns}
    L.append("columns: " + " | ".join(f"{c} [{types[c]}{', %d missing' % miss[c] if miss[c] else ''}]" for c in raw.columns))
    tgt_cands = [c for c in raw.columns if miss[c] > 0 and miss[c] < n] if not target else [target]
    if tgt_cands: L.append(f"missing-token columns (likely target / to predict): {[(c, miss[c], f'{miss[c]/n:.1%}') for c in tgt_cands]}")
    dup = int(raw.duplicated().sum()); L.append(f"duplicate rows: {dup:,} ({dup/n:.1%}); first col unique: {raw.iloc[:,0].is_unique}")
    d = raw.sample(sample, random_state=0) if n > sample else raw
    num = {}
    for c in raw.columns:
        t = types[c]; s = raw[c]
        if t == "numeric":
            v = pd.to_numeric(s.where(~s.isin(cio.MISSING)), errors="coerce"); bad = s[v.isna() & ~s.isin(cio.MISSING)]
            dirty = ""
            if len(bad):
                vc = bad.value_counts().head(6); parsed = sum(cio.word2num(x) is not None for x in bad.unique())
                dirty = f"  DIRTY: {len(bad)} non-numeric tokens { {k: int(v) for k, v in vc.items()} } ({parsed}/{bad.nunique()} unique parse as number words)"
            x = v.dropna()
            if len(x) == 0: continue
            q = x.quantile([.01, .25, .5, .75, .99]); iqr = q[.75] - q[.25]; out_n = int(((x < q[.25] - 3 * iqr) | (x > q[.75] + 3 * iqr)).sum()) if iqr > 0 else 0
            h, e = np.histogram(x, bins=12)
            L.append(f"- {c}: min {x.min():.4g} p1 {q[.01]:.4g} p25 {q[.25]:.4g} med {q[.5]:.4g} p75 {q[.75]:.4g} p99 {q[.99]:.4g} max {x.max():.4g} | mean {x.mean():.4g} sd {x.std():.4g} | uniq {x.nunique()} | 3xIQR outliers {out_n}\n    hist[{e[0]:.4g}..{e[-1]:.4g}] {_spark(h)} {h.tolist()}{dirty}")
        elif t == "list":
            toks = d[c].map(lambda z: str(z).split()); ln = toks.map(len); vocab = pd.Series([t_ for ts in toks.head(20000) for t_ in ts]).value_counts()
            first_last = toks.map(lambda ts: ts[0] == ts[-1]).mean(); pal = toks.map(lambda ts: ts == ts[::-1]).mean()
            L.append(f"- {c}: LIST len min {ln.min()} med {int(ln.median())} max {ln.max()} | vocab {len(vocab)} top { {k: int(v) for k, v in vocab.head(5).items()} } | first==last {first_last:.1%} | palindromic {pal:.1%} | distinct rows {raw[c].nunique():,}")
        elif t == "categorical":
            vc = raw[c].value_counts(dropna=False); L.append(f"- {c}: CAT {len(vc)} distinct; top { {k: int(v) for k, v in vc.head(8).items()} }")
    # structure: time / id / panel
    tcols = [c for c in raw.columns if TIME_HINT.match(c)]; icols = [c for c in raw.columns if ID_HINT.search(c) and types[c] == "numeric"]
    if tcols or icols:
        s = []
        for c in tcols[:1]:
            v = pd.to_numeric(raw[c], errors="coerce"); s.append(f"time col '{c}': {v.min():.0f}..{v.max():.0f}, {v.nunique()} distinct, monotonic {bool(v.is_monotonic_increasing)}")
        for c in icols[:2]:
            v = pd.to_numeric(raw[c], errors="coerce"); g = v.value_counts(); s.append(f"id col '{c}': {v.nunique():,} distinct, rows/id min {g.min()} med {int(g.median())} max {g.max()}, id range {v.min():.0f}..{v.max():.0f}")
        L.append("structure: " + "; ".join(s))
    # correlations (spearman) among numeric columns
    ncols = [c for c, t in types.items() if t == "numeric"]
    if len(ncols) >= 2:
        X = pd.DataFrame({c: pd.to_numeric(d[c].where(~d[c].isin(cio.MISSING)), errors="coerce") for c in ncols[:25]})
        C = X.corr(method="spearman"); pairs = [(abs(C.iloc[i, j]), C.iloc[i, j], C.columns[i], C.columns[j]) for i in range(len(C)) for j in range(i)]
        pairs = sorted(pairs, reverse=True)[:top_corr]
        L.append("top spearman pairs: " + ("; ".join(f"{a}~{b} {r:+.2f}" for _, r, a, b in pairs if abs(r) > 0.2) or "none > 0.2"))
        t = target or (tgt_cands[0] if tgt_cands and types.get(tgt_cands[0]) == "numeric" else None)
        if t and t in C.columns: L.append(f"spearman with target '{t}': " + ", ".join(f"{k} {v:+.2f}" for k, v in C[t].drop(t).sort_values(key=abs, ascending=False).items()))
    text = "\n".join(L)
    if out: Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(text, encoding="utf-8")
    print(text); return None

def compare(a, b, na="A", nb_="B", max_cols=30, adversarial=True):
    """Distribution shift between two frames (train vs test, regime 1 vs regime 2): per-column KS + mean/std ratio; adversarial AUC (0.5 = same)."""
    from scipy.stats import ks_2samp
    cols = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])][:max_cols]; rows = []
    for c in cols:
        x, y = a[c].dropna(), b[c].dropna()
        if len(x) < 5 or len(y) < 5: continue
        xs, ys = x.sample(min(len(x), 50000), random_state=0), y.sample(min(len(y), 50000), random_state=0)
        rows.append((c, ks_2samp(xs, ys).statistic, x.mean(), y.mean(), x.std(), y.std(), (y.mean() - x.mean()) / (x.std() + 1e-12)))
    df = pd.DataFrame(rows, columns=["col", "KS", f"mean_{na}", f"mean_{nb_}", f"sd_{na}", f"sd_{nb_}", "mean_shift_in_sd"]).sort_values("KS", ascending=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if adversarial and len(cols) >= 1:
        from sklearn.model_selection import cross_val_predict; from sklearn.metrics import roc_auc_score; import lightgbm as lgb
        A, B = a[cols].sample(min(len(a), 20000), random_state=0), b[cols].sample(min(len(b), 20000), random_state=0)
        X = pd.concat([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
        m = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, verbose=-1, n_jobs=2); p = cross_val_predict(m, X, y, cv=3, method="predict_proba")[:, 1]
        m.fit(X, y); imp = pd.Series(m.feature_importances_, cols).sort_values(ascending=False)
        print(f"adversarial AUC {roc_auc_score(y, p):.3f} (0.5 = indistinguishable, >0.7 = real shift); top drivers: { {k: int(v) for k, v in imp.head(5).items()} }")
    return df


def join_keys(frames: dict, min_overlap=0.2):
    """Find columns that look like join keys across frames {name: df}: same name or high value overlap. Prints coverage so merges are not blind."""
    names = list(frames); found = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = frames[names[i]], frames[names[j]]
            for ca in A.columns:
                if not pd.api.types.is_integer_dtype(A[ca]) and not (A[ca].dtype == float and (A[ca].dropna() % 1 == 0).all()): continue
                for cb in B.columns:
                    if not pd.api.types.is_integer_dtype(B[cb]) and not (B[cb].dtype == float and (B[cb].dropna() % 1 == 0).all()): continue
                    sa, sb = set(A[ca].dropna().unique()), set(B[cb].dropna().unique())
                    if len(sa) < 5 or len(sb) < 5: continue
                    ov = len(sa & sb) / min(len(sa), len(sb))
                    if ov >= min_overlap and (ca == cb or ov > 0.8): found.append((names[i], ca, names[j], cb, len(sa), len(sb), ov))
    for f in found: print(f"{f[0]}.{f[1]} ({f[4]:,} uniq) ~ {f[2]}.{f[3]} ({f[5]:,} uniq): overlap {f[6]:.0%}")
    if not found: print("no join keys found")
    return found
