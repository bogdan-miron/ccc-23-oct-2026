#!/usr/bin/env python3
"""Cross-platform env bootstrap (Linux, macOS, Windows). Usage: python setup_env.py [venv|torch|core|extra|verify|all]
Stages are idempotent and resumable. In a sandbox with a 180 s per-call cap run them in SEPARATE calls (torch ~40 s, core ~2 min, extra ~2 min).
Env lives OUTSIDE the repo (cloud-sync / FUSE friendly): $CF_VENV, default ~/cfenv. torch is the CPU build (no GPU assumed)."""
import os, shutil, subprocess, sys, venv as _venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = Path(os.environ.get("CF_VENV", Path.home() / "cfenv"))
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

def sh(cmd, **kw):
    cmd = [str(c) for c in cmd]; print("+", " ".join(cmd), flush=True); subprocess.check_call(cmd, **kw)

_uv = None
def uv():
    """Return a command prefix for uv (fast installer) or None to fall back to plain pip."""
    global _uv
    if _uv is not None: return _uv or None
    exe = shutil.which("uv")
    if not exe:
        for extra in ([], ["--break-system-packages"], ["--user"]):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv", *extra], stderr=subprocess.DEVNULL); break
            except Exception: continue
        exe = shutil.which("uv")
    _uv = [exe] if exe else ([sys.executable, "-m", "uv"] if subprocess.call([sys.executable, "-m", "uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0 else [])
    return _uv or None

def ensure_venv():
    if PY.exists(): return
    u = uv()
    if u: sh(u + ["venv", VENV, "--python", sys.executable, "-q"])
    else: print("uv unavailable, using stdlib venv + pip (slower)"); _venv.create(VENV, with_pip=True)

def install(args, torch=False):
    u = uv()
    if u:
        extra = ["--index-url", TORCH_INDEX, "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match"] if torch else []
        sh(u + ["pip", "install", "--python", PY, "-q", *args, *extra])
    else:
        extra = ["--index-url", TORCH_INDEX, "--extra-index-url", "https://pypi.org/simple"] if torch else []
        sh([PY, "-m", "pip", "install", "-q", *args, *extra])

def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in ("venv", "torch", "core", "extra", "verify", "all"): sys.exit(__doc__)
    ensure_venv()
    if stage in ("torch", "all"): install(["torch", "torchvision"], torch=True)
    if stage in ("core", "all"): install(["-r", ROOT / "requirements.txt"])
    if stage in ("extra", "all"): install(["-r", ROOT / "requirements-extra.txt"])
    if stage in ("verify", "all"):
        env = dict(os.environ, PYTHONPATH=str(ROOT)); sh([PY, "-I", ROOT / "tools" / "verify_env.py"], env=env)
    print(f"setup_env stage={stage} done; venv={VENV}")

if __name__ == "__main__":
    main()
