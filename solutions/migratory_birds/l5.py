"""migratory-birds level 5. Top 50 BOPs by Arrivals for each of the next 30 days.
Structure found in EDA: Arrivals = Occupancy x BOP factor x day factor x slowly varying per-BOP deviation. The day factor cancels in a within-day ranking.
Panel arrays: A, O [days, 2500] (days 1..760, A missing from 731), W [days, 2] wind, I [2500] insects delta."""
from pathlib import Path
import numpy as np, pandas as pd

NB = 2500; H = 30
FEATS = ["h", "lo", "lo_prev", "lo_next", "d_prev", "d_next", "lo_rel", "lo_rank", "lo_c", "drift", "v_all", "v60", "dev3", "dev7", "dev14",
         "lA3", "lA7", "lA30", "ins", "wx", "wy", "wx3", "wy3"]

def load_panel(path):
    df = pd.read_csv(path, na_values=["missing"], dtype={"Day": "int16", "BOP": "int16"}, skipinitialspace=True)
    df.columns = ["day", "bop", "arr", "occ", "wx", "wy", "ins"]
    nd = int(df.day.max())
    A = df.pivot(index="day", columns="bop", values="arr").values.astype("float32"); O = df.pivot(index="day", columns="bop", values="occ").values.astype("float32")
    W = df.groupby("day")[["wx", "wy"]].first().values.astype("float32"); I = df.groupby("bop").ins.first().values.astype("float32")
    return A, O, W, I

class Panel:
    def __init__(self, A, O, W, I):
        self.A, self.O, self.W, self.I = A, O, W, I; nd = len(O)
        self.lO = np.log(O).astype("float32")
        known = np.isfinite(A).all(1); self.nknown = int(known.sum())     # days with Arrivals (730 at test time)
        la = np.log1p(A[:self.nknown]); self.la = la
        Z = la - self.lO[:self.nknown]; self.Z = (Z - Z.mean(1, keepdims=True)).astype("float32")
        self.cs = np.vstack([np.zeros((1, NB)), np.cumsum(self.Z, 0)]).astype("float64")
        self.lac = np.vstack([np.zeros((1, NB)), np.cumsum(la - la.mean(1, keepdims=True), 0)]).astype("float64")

    def mean_last(self, cs, c, k): return ((cs[c] - cs[max(0, c - k)]) / min(k, c)).astype("float32")

    def lowrank(self, c, k=4):
        """Top-k right singular vectors of the BOP-demeaned log-ratio matrix before day c (k x 2500)."""
        from scipy.sparse.linalg import svds
        Zc = self.Z[:c] - self.Z[:c].mean(0); _, _, Vt = svds(Zc.astype("float32"), k=k); return Vt

    def features(self, c, days=None):
        """Feature tensor [len(days), 2500, F] for target days (0-based indices, default c..c+29) using Arrivals only before index c."""
        days = np.arange(c, c + H) if days is None else np.asarray(days); n = len(days); F = np.zeros((n, NB, len(FEATS)), np.float32)
        v_all = self.mean_last(self.cs, c, c); v60 = self.mean_last(self.cs, c, 60)
        dev3 = self.mean_last(self.cs, c, 3) - v_all; dev7 = self.mean_last(self.cs, c, 7) - v_all; dev14 = self.mean_last(self.cs, c, 14) - v_all
        lA3, lA7, lA30 = (self.mean_last(self.lac, c, k) for k in (3, 7, 30))
        nd = len(self.O)
        for i, t in enumerate(days):
            lo = self.lO[t]; lop = self.lO[t - 1]; lon = self.lO[t + 1] if t + 1 < nd else lo   # last forecast day has no next-day occupancy: neutral fill
            rank = lo.argsort().argsort() / (NB - 1.0)
            w3 = self.W[max(0, t - 2):t + 1].mean(0)
            cols = [np.full(NB, t - c + 1, np.float32), lo, lop, lon, lo - lop, lon - lo, lo - lo.mean(), rank, self.lO[c - 1], lo - self.lO[c - 1],
                    v_all, v60, dev3, dev7, dev14, lA3, lA7, lA30, self.I, np.full(NB, self.W[t, 0]), np.full(NB, self.W[t, 1]), np.full(NB, w3[0]), np.full(NB, w3[1])]
            F[i] = np.stack(cols, 1)
        return F

    def target(self, c, days=None):
        days = np.arange(c, c + H) if days is None else np.asarray(days); la = np.log1p(self.A[days]); return (la - la.mean(1, keepdims=True)).astype("float32")

def top50(score): return np.argpartition(-score, 50, axis=-1)[..., :50]
def overlap(pred_scores, true_rows):
    P = top50(pred_scores); T = top50(true_rows); return float(np.mean([len(set(p) & set(t)) / 50 for p, t in zip(P, T)]))


HEADER = "Day,Top 50 Arrivals BOPs"

def fit_ensemble(P, last_cut, seeds=(0, 1, 2), frac=0.15, step=5, first_cut=30):
    """Train on (day, BOP) rows from cutoffs first_cut..last_cut (targets up to day last_cut+30). Target: within-day centered log1p(Arrivals).
    Features are computed once per cutoff; each seed draws its own row subsample."""
    import lightgbm as lgb
    rngs = {sd: np.random.default_rng(sd) for sd in seeds}; Xs = {sd: [] for sd in seeds}; ys = {sd: [] for sd in seeds}
    for c in range(first_cut, last_cut + 1, step):
        F = P.features(c); y = P.target(c)
        for sd in seeds:
            m = rngs[sd].random(y.shape) < frac; Xs[sd].append(F[m]); ys[sd].append(y[m])
    return [lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              n_jobs=2, random_state=sd, verbose=-1).fit(np.concatenate(Xs[sd]), np.concatenate(ys[sd])) for sd in seeds]

def predict_scores(P, models, c):
    F = P.features(c); X = F.reshape(-1, F.shape[-1]); return np.mean([m.predict(X) for m in models], 0).reshape(F.shape[0], NB)

def backtest(A, O, W, I, c, seeds=(0, 1, 2)):
    """Mask Arrivals from day index c on, run the exact pipeline, score on the real days c..c+29."""
    Am = A.copy(); Am[c:] = np.nan; P = Panel(Am, O, W, I); models = fit_ensemble(P, c - H, seeds)
    P.A = A; return overlap(predict_scores(P, models, c), A[c:c + H])

_cache = {}
def solve(path, ctx) -> str:
    key = str(path)
    if key not in _cache:
        A, O, W, I = load_panel(path); P = Panel(A, O, W, I); c = P.nknown
        models = fit_ensemble(P, c - H); S = predict_scores(P, models, c); _cache[key] = (S, c)
    S, c = _cache[key]; lines = [HEADER]
    for i in range(S.shape[0]):
        order = np.argsort(-S[i])[:50] + 1; lines.append(f"{c + 1 + i},{' '.join(map(str, order))}")
    return "\n".join(lines)
