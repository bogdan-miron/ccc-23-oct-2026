"""Inventory the challenge folders. Unzips new level zips. Usage: ./run.sh tools/scan.py [challenge] [--upto N]
--upto N (dry run, blind mode): only show levels 1..N and remember N in notes/<challenge>/.unlocked; later calls honor it. Delete that file to see everything.
Shows per level: pdf, inputs (sizes, rows, header), examples with expected outputs, extras. Flags what is new vs notes/<challenge>/.scan.json."""
import json, re, sys, zipfile
from pathlib import Path
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from cf import paths

def unzip_all(base: Path):
    for z in sorted(base.rglob("*.zip")):
        m = re.search(r"level[-_ ]?(\d+)", z.name, re.I)
        dest = (base / f"level-{m.group(1)}") if m and z.parent == base else z.with_suffix("")
        if dest.exists() and any(dest.iterdir()): continue
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist(); top = {n.split("/")[0] for n in names if n.strip("/")}
            zf.extractall(dest)
            if len(top) == 1 and all(n.startswith(next(iter(top)) + "/") or n == next(iter(top)) + "/" for n in names) and (dest / next(iter(top))).is_dir():
                inner = dest / next(iter(top))
                for f in inner.iterdir(): f.rename(dest / f.name)
                inner.rmdir()
        print(f"[unzip] {z.name} -> {dest.relative_to(paths.ROOT) if dest.is_relative_to(paths.ROOT) else dest}")

def head(p, n=1):
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f: return [next(f).rstrip() for _ in range(n)]
    except StopIteration: return []

def nlines(p):
    with open(p, "rb") as f: return sum(1 for _ in f)

def main():
    argv = sys.argv[1:]; upto = None
    if "--upto" in argv:
        i = argv.index("--upto"); upto = int(argv[i + 1]); del argv[i:i + 2]
    chs = [paths.Challenge(argv[0])] if argv else paths.find_challenges()
    if not chs: print("no challenge folders found (need <challenge>/level-N/...)"); return
    for ch in chs:
        unzip_all(ch.dir)
        seen_f = ch.dir.parent / ".." / "notes" / ch.name / ".scan.json"; seen_f = paths.ROOT / "notes" / ch.name / ".scan.json"
        seen = json.loads(seen_f.read_text()) if seen_f.exists() else {}
        ufile = paths.ROOT / "notes" / ch.name / ".unlocked"
        if upto is not None: ufile.parent.mkdir(parents=True, exist_ok=True); ufile.write_text(str(upto))
        lim = int(ufile.read_text()) if ufile.exists() else None
        lvls = [l for l in ch.levels() if lim is None or l.n <= lim]
        print(f"# {ch.name}  ({len(lvls)} levels shown" + (f", blind mode: unlocked up to {lim}" if lim else "") + ")"); cur = {}
        for lv in lvls:
            tag = " NEW" if str(lv.n) not in seen else ""
            print(f"\n## level {lv.n}{tag}: pdf={lv.pdf.name if lv.pdf else None}")
            for f in lv.inputs:
                ex = lv.expected(f); sz = f.stat().st_size
                print(f"  in  {f.name:34s} {sz/1e3:10.1f} KB {nlines(f):>9,} lines  hdr: {head(f)[0][:70] if sz else ''}" + (f"   [expected out: {ex.name}]" if ex else ""))
            for f in lv.extras: print(f"  ext {f.name:34s} {f.stat().st_size/1e3:10.1f} KB {nlines(f):>9,} lines  hdr: {(head(f) or [''])[0][:70]}")
            cur[str(lv.n)] = [f.name for f in lv.inputs]
            outd = lv.out_dir
            if outd.exists(): print(f"  our outputs: {sorted(p.name for p in outd.glob('out_*'))}")
        seen_f.parent.mkdir(parents=True, exist_ok=True); seen_f.write_text(json.dumps(cur))

if __name__ == "__main__":
    main()
