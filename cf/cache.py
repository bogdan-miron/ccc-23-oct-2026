"""Disk cache for expensive steps (parse big CSV, features, fitted models) so each tool call can restart cheaply.
@disk_cache  -> keyed by function source + args (Path args keyed by size+mtime). Stored under $CF_STATE/cache (default ~/.cfstate/cache)."""
import hashlib, inspect, os, pickle
from pathlib import Path

def _root():
    d = Path(os.environ.get("CF_STATE", Path.home() / ".cfstate")) / "cache"; d.mkdir(parents=True, exist_ok=True); return d

def _key(fn, args, kw):
    h = hashlib.sha1(inspect.getsource(fn).encode())
    for a in list(args) + [v for _, v in sorted(kw.items())]:
        if isinstance(a, Path) and a.exists(): h.update(f"{a}:{a.stat().st_size}:{int(a.stat().st_mtime)}".encode())
        else: h.update(repr(a).encode() if not hasattr(a, "shape") else f"{type(a)}{a.shape}{hash(a.tobytes()) if hasattr(a, 'tobytes') else ''}".encode())
    return h.hexdigest()[:16]

def disk_cache(fn):
    def wrapper(*a, **kw):
        f = _root() / f"{fn.__name__}-{_key(fn, a, kw)}.pkl"
        if f.exists():
            with open(f, "rb") as fh: return pickle.load(fh)
        r = fn(*a, **kw)
        with open(f, "wb") as fh: pickle.dump(r, fh, protocol=5)
        return r
    wrapper.__name__ = fn.__name__; wrapper.__wrapped__ = fn; return wrapper
