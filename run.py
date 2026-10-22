#!/usr/bin/env python3
"""Run python inside the project env with the repo importable (all platforms).
Usage: python run.py script.py args | python run.py -c "code" | python run.py -m pytest -q tests   (./run.sh does the same on Linux/macOS/WSL)"""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = Path(os.environ.get("CF_VENV", Path.home() / "cfenv"))
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not PY.exists():
    sys.exit(f"env missing at {VENV}: run `python setup_env.py venv`, then `torch`, `core`, (`extra`), `verify` (separate calls if the shell has a short time cap)")
env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
env.setdefault("PYTHONUNBUFFERED", "1"); env.setdefault("MPLBACKEND", "Agg")
env.setdefault("OMP_NUM_THREADS", str(min(os.cpu_count() or 2, 8)))
if os.name == "nt": sys.exit(subprocess.call([str(PY), *sys.argv[1:]], env=env))
os.execve(str(PY), [str(PY), *sys.argv[1:]], env)
