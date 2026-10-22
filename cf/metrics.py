"""Metrics registry. get('rmse') -> (fn, higher_is_better). Top-K overlap is the 'accuracy' used by ranking levels (|pred ∩ true| / K)."""
import numpy as np
from sklearn import metrics as skm

def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))
def mae(y, p): return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))
def r2(y, p): return float(skm.r2_score(y, p))
def accuracy(y, p): return float(skm.accuracy_score(y, p))
def f1_macro(y, p): return float(skm.f1_score(y, p, average="macro"))
def f1_micro(y, p): return float(skm.f1_score(y, p, average="micro"))
def f1_weighted(y, p): return float(skm.f1_score(y, p, average="weighted"))
def logloss(y, p): return float(skm.log_loss(y, p))
def auc(y, p): return float(skm.roc_auc_score(y, p, multi_class="ovr") if np.ndim(p) == 2 and np.shape(p)[1] > 2 else skm.roc_auc_score(y, p))
def spearman(y, p):
    from scipy.stats import spearmanr; return float(spearmanr(y, p)[0])

def topk_overlap(pred, true, k=None):
    """Mean over groups of |set(pred) ∩ set(true)| / k. pred/true: list of iterables (one per day/query)."""
    s = []
    for P, T in zip(pred, true):
        P, T = list(P), list(T); kk = k or len(T); s.append(len(set(P[:kk]) & set(T[:kk])) / kk)
    return float(np.mean(s))

def ndcg_at_k(rel_true, score_pred, k=50):
    """rel_true, score_pred: 2D arrays [queries, items]."""
    return float(skm.ndcg_score(np.atleast_2d(rel_true), np.atleast_2d(score_pred), k=k))

REGISTRY = {"rmse": (rmse, False), "mae": (mae, False), "r2": (r2, True), "accuracy": (accuracy, True), "f1_macro": (f1_macro, True),
            "f1_micro": (f1_micro, True), "f1_weighted": (f1_weighted, True), "logloss": (logloss, False), "auc": (auc, True), "spearman": (spearman, True)}
NEEDS_PROBA = {"logloss", "auc"}

def get(name):
    if callable(name): return name, True
    return REGISTRY[name]
