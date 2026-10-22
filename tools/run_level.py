"""Run a level's solver on its input files, write outputs, validate against examples + format. Usage:
  ./run.sh tools/run_level.py <challenge> <N> [--only a,b] [--no-example] [--freeze] [--regress]
Writes submissions/<challenge>/level-N/out_*. Example inputs MUST reproduce the organizer's expected output exactly (hard fail otherwise).
--freeze  : copy current outputs to .../passed/ (do this after the user confirms the upload passed)
--regress : re-run all earlier levels' examples + compare with frozen passed outputs (guards the snowball when shared code changed)."""
import argparse, importlib, shutil, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from cf import paths, submit

def _style_ref(ch, n):
    """Levels without an example out file (L5, L6): copy the CRLF/newline style of the nearest earlier passed output."""
    for l in sorted(ch.levels(), key=lambda l: -l.n):
        if l.n < n and (l.out_dir / "passed").exists():
            c = sorted((l.out_dir / "passed").glob("out_*"))
            if c: return c[0]
    return None

def run(ch, n, only=None, check_example=True, quiet=False):
    lv = ch.level(n); mod = importlib.import_module(f"solutions.{ch.module}.l{n}"); importlib.reload(mod)
    ctx = SimpleNamespace(challenge=ch, level=lv, root=paths.ROOT); rows = []
    for f in lv.inputs:
        tag = f.stem.split("_", 2)[-1]
        if only and not any(o in f.name for o in only): continue
        t = time.time(); status = "ok"; msg = ""
        try:
            txt = mod.solve(f, ctx); ex = lv.expected(f)
            outp = submit.write_text(lv.out_path(f), txt, like=ex or next((lv.expected(g) for g in lv.inputs if lv.expected(g)), None) or _style_ref(ch, n))  # non-example files copy the style of the level example
            if ex and check_example:
                ok, m = submit.diff_example(outp, ex); status = "EXAMPLE OK" if ok else "EXAMPLE MISMATCH"; msg = "" if ok else m
            probs = submit.check_output(outp, example_out=ex or next((lv.expected(g) for g in lv.inputs if lv.expected(g)), None))
            if probs: status += " | FORMAT: " + "; ".join(probs)
        except Exception as e:
            status = "ERROR"; msg = traceback.format_exc(limit=4)
        rows.append((f.name, status, time.time() - t, msg))
    if not quiet:
        for name, st, secs, msg in rows: print(f"{name:36s} {st:28s} {secs:6.1f}s" + (f"\n{msg}" if msg else ""))
        if any(r[1].startswith(("ERROR", "EXAMPLE MISMATCH")) or "FORMAT:" in r[1] for r in rows): print("BLOCKED: do not hand these files to the user until the failures above are fixed.")
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("challenge"); ap.add_argument("n", type=int); ap.add_argument("--only"); ap.add_argument("--no-example", action="store_true")
    ap.add_argument("--freeze", action="store_true"); ap.add_argument("--regress", action="store_true"); a = ap.parse_args()
    ch = paths.Challenge(a.challenge); lv = ch.level(a.n)
    if a.freeze:
        dst = lv.out_dir / "passed"; dst.mkdir(exist_ok=True)
        for f in lv.out_dir.glob("out_*"): shutil.copy(f, dst / f.name)
        print(f"frozen {len(list(dst.glob('out_*')))} files -> {dst}"); return
    if a.regress:
        bad = 0
        for l in ch.levels():
            if l.n >= a.n or not (l.out_dir / "passed").exists(): continue
            rows = run(ch, l.n, quiet=True)
            for f in l.inputs:
                fz = l.out_dir / "passed" / l.out_path(f).name
                if fz.exists():
                    cur = l.out_path(f); same = cur.exists() and submit._norm(cur.read_text()) == submit._norm(fz.read_text())
                    if not same: bad += 1; print(f"REGRESSION level {l.n} {f.name}: output differs from frozen passed version")
        print("regression: all earlier passed outputs unchanged" if not bad else f"{bad} regressions"); sys.exit(1 if bad else 0)
    rows = run(ch, a.n, a.only.split(",") if a.only else None, not a.no_example)
    sys.exit(1 if any(r[1].startswith(("ERROR", "EXAMPLE MISMATCH")) for r in rows) else 0)

if __name__ == "__main__":
    main()
