"""Model factory with sane FAST defaults (CPU, CF_JOBS threads). make('lgbm','reg'|'clf', **overrides). No tuning here on purpose:
if defaults are far from the target, fix FEATURES / problem understanding, not hyper-parameters."""
import numpy as np
from cf.config import NJOBS

def _torch_mlp():
    import torch, torch.nn as nn
    from sklearn.base import BaseEstimator
    class TorchMLP(BaseEstimator):
        """Small standardizing MLP with sklearn API. task='reg'|'clf'."""
        def __init__(self, task="reg", hidden=(128, 64), epochs=60, lr=2e-3, batch=256, wd=1e-4, drop=0.1, seed=0):
            self.task, self.hidden, self.epochs, self.lr, self.batch, self.wd, self.drop, self.seed = task, hidden, epochs, lr, batch, wd, drop, seed
        def fit(self, X, y):
            torch.manual_seed(self.seed); X = np.asarray(X, np.float32); self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
            Xt = torch.tensor((X - self.mu_) / self.sd_)
            if self.task == "clf":
                self.classes_, yi = np.unique(y, return_inverse=True); yt = torch.tensor(yi); out = len(self.classes_)
            else:
                y = np.asarray(y, np.float32); self.ym_, self.ys_ = y.mean(), y.std() + 1e-6; yt = torch.tensor((y - self.ym_) / self.ys_)[:, None]; out = 1
            layers, d = [], X.shape[1]
            for h in self.hidden: layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.drop)]; d = h
            self.net_ = nn.Sequential(*layers, nn.Linear(d, out)); opt = torch.optim.AdamW(self.net_.parameters(), self.lr, weight_decay=self.wd)
            lossf = nn.CrossEntropyLoss() if self.task == "clf" else nn.MSELoss(); n = len(Xt)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr * 3, total_steps=self.epochs * ((n + self.batch - 1) // self.batch))
            self.net_.train()
            for _ in range(self.epochs):
                perm = torch.randperm(n)
                for i in range(0, n, self.batch):
                    b = perm[i:i + self.batch]; opt.zero_grad(); l = lossf(self.net_(Xt[b]), yt[b]); l.backward(); opt.step(); sched.step()
            return self
        def _fwd(self, X):
            self.net_.eval()
            with torch.no_grad(): return self.net_(torch.tensor((np.asarray(X, np.float32) - self.mu_) / self.sd_))
        def predict(self, X):
            o = self._fwd(X)
            return self.classes_[o.argmax(1).numpy()] if self.task == "clf" else o.numpy().ravel() * self.ys_ + self.ym_
        def predict_proba(self, X): return torch.softmax(self._fwd(X), 1).numpy()
    return TorchMLP

def make(name, task="reg", seed=0, **kw):
    reg = task == "reg"
    if name in ("ridge", "linear", "lasso", "enet"):
        from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
        from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
        if not reg: raise ValueError("use logreg")
        return make_pipeline(StandardScaler(), {"ridge": Ridge, "linear": LinearRegression, "lasso": Lasso, "enet": ElasticNet}[name](**kw))
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **kw))
    if name == "knn":
        from sklearn.neighbors import KNeighborsRegressor as R, KNeighborsClassifier as C
        from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), (R if reg else C)(**{"n_neighbors": 7, **kw}))
    if name in ("rf", "et"):
        from sklearn import ensemble as E
        cls = {("rf", True): E.RandomForestRegressor, ("rf", False): E.RandomForestClassifier, ("et", True): E.ExtraTreesRegressor, ("et", False): E.ExtraTreesClassifier}[(name, reg)]
        return cls(**{"n_estimators": 300, "n_jobs": NJOBS, "random_state": seed, **kw})
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor as R, HistGradientBoostingClassifier as C
        return (R if reg else C)(**{"random_state": seed, **kw})
    if name == "lgbm":
        import lightgbm as lgb
        return (lgb.LGBMRegressor if reg else lgb.LGBMClassifier)(**{"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "n_jobs": NJOBS, "random_state": seed, "verbose": -1, **kw})
    if name == "xgb":
        import xgboost as xgb
        return (xgb.XGBRegressor if reg else xgb.XGBClassifier)(**{"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "n_jobs": NJOBS, "random_state": seed, **kw})
    if name == "cat":
        import catboost as cb
        return (cb.CatBoostRegressor if reg else cb.CatBoostClassifier)(**{"iterations": 500, "learning_rate": 0.08, "depth": 6, "verbose": 0, "thread_count": NJOBS, "random_seed": seed, "allow_writing_files": False, **kw})
    if name == "mlp":
        return _torch_mlp()(task=task, seed=seed, **kw)
    raise ValueError(f"unknown model {name}")

REG_SUITE = ["ridge", "knn", "rf", "lgbm", "cat"]
CLF_SUITE = ["logreg", "knn", "rf", "lgbm", "cat"]
