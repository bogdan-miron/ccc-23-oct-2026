"""Checkpoint/restore of notebook variables between tool calls (processes die when a call ends, so there is no live kernel).
Each variable is pickled separately with dill; unpicklable ones are reported, never fatal. Names starting with '_' are NOT saved (use for scratch)."""
import time, shutil
from pathlib import Path
import dill

SKIP = {"In", "Out", "get_ipython", "exit", "quit", "open"}

def _fmt(b): return f"{b/1e6:.0f}MB" if b >= 1e6 else f"{b/1e3:.0f}KB"

def checkpoint(dirpath, ns, max_bytes=1_500_000_000, quiet=False):
    d = Path(dirpath); d.mkdir(parents=True, exist_ok=True); t0 = time.time(); saved, failed, big, tot = [], [], [], 0
    names = set()
    try:
        from cf import nbinit; base = vars(nbinit)
    except Exception: base = {}
    for k, v in list(ns.items()):
        if k.startswith("_") or k in SKIP or (k in base and base[k] is v): continue
        try:
            b = dill.dumps(v, protocol=5)
        except Exception as e:
            failed.append(f"{k} ({type(e).__name__})"); continue
        if len(b) > max_bytes: big.append(k); continue
        (d / f"{k}.pkl").write_bytes(b); names.add(k); saved.append(k); tot += len(b)
    for f in d.glob("*.pkl"):
        if f.stem not in names: f.unlink()
    if not quiet:
        print(f"[state] saved {len(saved)} vars ({_fmt(tot)}) in {time.time()-t0:.1f}s" + (f"; NOT persisted (unpicklable): {failed}" if failed else "") + (f"; too big: {big}" if big else ""))
    return saved, failed

def restore(dirpath, ns, quiet=False):
    d = Path(dirpath)
    if not d.exists(): return 0
    t0 = time.time(); n = tot = 0; failed = []
    for f in sorted(d.glob("*.pkl")):
        try:
            ns[f.stem] = dill.loads(f.read_bytes()); n += 1; tot += f.stat().st_size
        except Exception as e:
            failed.append(f"{f.stem} ({type(e).__name__})")
    if not quiet: print(f"[state] restored {n} vars ({_fmt(tot)}) in {time.time()-t0:.1f}s" + (f"; FAILED to restore: {failed}" if failed else ""))
    return n
