"""Cross-validation + quick model bake-off. Pick the split that mimics the test setup: kfold (iid), stratified (imbalanced clf),
group (flocks/users/BOPs must not leak across folds), time (forecasting: train on past, validate on future)."""
import time as _time
from dataclasses import dataclass, field
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from cf import metrics as M, models as MD

@dataclass
class CVResult:
    oof: np.ndarray; score: float; folds: list; metric: str; secs: float; models: list = field(default_factory=list); mask: np.ndarray = None
    def __repr__(self): return f"CV[{self.metric}] {self.score:.4f} +- {np.std(self.folds):.4f} folds={np.round(self.folds, 4).tolist()} ({self.secs:.0f}s)"

def make_splits(n, y=None, kind="kfold", n_splits=5, groups=None, time=None, seed=0):
    idx = np.arange(n)
    if kind == "kfold": return list(KFold(n_splits, shuffle=True, random_state=seed).split(idx))
    if kind == "stratified": return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(idx, y))
    if kind == "group": return list(GroupKFold(n_splits).split(idx, y, groups))
    if kind == "time":
        order = np.argsort(np.asarray(time), kind="stable"); chunks = np.array_split(order, n_splits + 1)
        return [(np.concatenate(chunks[:i + 1]), chunks[i + 1]) for i in range(n_splits)]
    raise ValueError(kind)

def cross_val(model_fn, X, y, task="reg", metric=None, kind=None, n_splits=5, groups=None, time=None, seed=0, fit_kw=None, keep_models=False, verbose=True):
    """model_fn: () -> fresh estimator (sklearn API). Returns CVResult with out-of-fold preds (NaN where never validated, e.g. time split)."""
    X = X.reset_index(drop=True) if hasattr(X, "reset_index") else np.asarray(X); y = np.asarray(y)
    metric = metric or ("rmse" if task == "reg" else "f1_macro"); fn, _ = M.get(metric); proba = metric in M.NEEDS_PROBA
    kind = kind or ("kfold" if task == "reg" else "stratified")
    splits = make_splits(len(y), y, kind, n_splits, groups, time, seed); t0 = _time.time()
    oof = np.full(len(y), np.nan, dtype=object if task == "clf" and not proba else float) if not proba else None
    folds, models = [], []
    for tr, va in splits:
        m = model_fn(); Xtr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]; Xva = X.iloc[va] if hasattr(X, "iloc") else X[va]
        m.fit(Xtr, y[tr], **(fit_kw or {}))
        if proba:
            p = m.predict_proba(Xva); oof = np.zeros((len(y), p.shape[1])) if oof is None else oof; oof[va] = p
            folds.append(fn(y[va], p if p.shape[1] > 2 or metric == "logloss" else p[:, 1]))
        else:
            p = m.predict(Xva); oof[va] = np.ravel(p); folds.append(fn(y[va], np.ravel(p)))
        if keep_models: models.append(m)
    mask = ~pd.isna(oof) if not proba else None
    res = CVResult(oof, float(np.mean(folds)), folds, metric, _time.time() - t0, models, mask)
    if verbose: print(res)
    return res

def bakeoff(X, y, task="reg", metric=None, names=None, time_limit=120, **kw):
    """Run default models, print leaderboard. Use FIRST on any tabular problem to know what a 'normal' score is (<1-2 min)."""
    names = names or (MD.REG_SUITE if task == "reg" else MD.CLF_SUITE); rows = []; t0 = _time.time()
    for nm in names:
        if _time.time() - t0 > time_limit: rows.append((nm, np.nan, np.nan, 0)); continue
        try:
            r = cross_val(lambda nm=nm: MD.make(nm, task), X, y, task, metric, verbose=False, **kw); rows.append((nm, r.score, float(np.std(r.folds)), r.secs))
        except Exception as e:
            rows.append((nm, np.nan, np.nan, 0)); print(f"[bakeoff] {nm} failed: {type(e).__name__}: {str(e)[:80]}")
    _, hib = M.get(metric or ("rmse" if task == "reg" else "f1_macro"))
    df = pd.DataFrame(rows, columns=["model", "score", "std", "secs"]).sort_values("score", ascending=not hib).reset_index(drop=True)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}")); return df
