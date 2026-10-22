"""Smoke-test the ML env. Prints one line per check. Run: ./setup_env.sh verify  (or ./run.sh tools/verify_env.py)"""
import importlib, time, sys, platform, os

MODS = ["numpy","pandas","polars","pyarrow","scipy","sklearn","statsmodels","numba","duckdb","lightgbm","xgboost","catboost",
        "optuna","imblearn","networkx","sympy","matplotlib","seaborn","torch","torchvision","IPython",
        "ipykernel","jupyter_client","nbformat","nbclient","dill","pymupdf","pdfplumber","yaml","tqdm","joblib","rich"]
EXTRA = ["ortools","rapidfuzz","transformers","sentence_transformers","cv2","PIL","skimage","timm","gymnasium","stable_baselines3"]  # optional stage: setup_env.py extra
bad = []
print(f"python {platform.python_version()} cpus={os.cpu_count()}")
for m in MODS:
    t = time.time()
    try:
        mod = importlib.import_module(m)
        print(f"ok   {m:22s} {getattr(mod,'__version__','')}  ({time.time()-t:.1f}s)")
    except Exception as e:
        bad.append(m); print(f"FAIL {m:22s} {type(e).__name__}: {str(e)[:80]}")

for m in EXTRA:
    try:
        mod = importlib.import_module(m); print(f"ok   {m:22s} {getattr(mod,'__version__','')}  (extra)")
    except Exception as e:
        print(f"skip {m:22s} extra not installed ({type(e).__name__})")

def timed(name, fn):
    t = time.time()
    try:
        r = fn(); print(f"ok   fit:{name:14s} {r}  ({time.time()-t:.1f}s)")
    except Exception as e:
        bad.append("fit:"+name); print(f"FAIL fit:{name:14s} {type(e).__name__}: {str(e)[:100]}")

try:
    import numpy as np
    from sklearn.datasets import make_regression, make_classification
    Xr, yr = make_regression(2000, 20, noise=5, random_state=0)
    Xc, yc = make_classification(2000, 20, n_classes=3, n_informative=8, random_state=0)
    from sklearn.metrics import r2_score, accuracy_score
    def lgb():
        import lightgbm as l; m=l.LGBMRegressor(n_estimators=100, verbose=-1).fit(Xr,yr); return f"r2={r2_score(yr,m.predict(Xr)):.3f}"
    def xgb():
        import xgboost as x; m=x.XGBClassifier(n_estimators=50).fit(Xc,yc); return f"acc={accuracy_score(yc,m.predict(Xc)):.3f}"
    def cat():
        from catboost import CatBoostClassifier as C; m=C(iterations=100, verbose=0, thread_count=2, allow_writing_files=False).fit(Xc,yc); return f"acc={accuracy_score(yc,m.predict(Xc).ravel()):.3f}"
    def sk():
        from sklearn.ensemble import RandomForestClassifier as R; m=R(100, n_jobs=2).fit(Xc,yc); return f"acc={m.score(Xc,yc):.3f}"
    def tch():
        import torch; torch.manual_seed(0); X=torch.tensor(Xr,dtype=torch.float32); y=torch.tensor(yr,dtype=torch.float32)[:,None]
        net=torch.nn.Sequential(torch.nn.Linear(20,32),torch.nn.ReLU(),torch.nn.Linear(32,1)); opt=torch.optim.Adam(net.parameters(),1e-2)
        for _ in range(100): opt.zero_grad(); l=torch.nn.functional.mse_loss(net(X),y); l.backward(); opt.step()
        return f"mse={l.item():.1f} threads={torch.get_num_threads()} cuda={torch.cuda.is_available()}"
    for n,f in [("lightgbm",lgb),("xgboost",xgb),("catboost",cat),("sklearn_rf",sk),("torch",tch)]: timed(n,f)
except Exception as e:
    print("FAIL setup of fit tests", e); bad.append("fit-setup")

print("FAILED:", bad if bad else "none")
sys.exit(1 if bad else 0)
