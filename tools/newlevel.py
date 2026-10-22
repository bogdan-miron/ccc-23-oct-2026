"""Scaffold one level: notes dir + SPEC template + solver stub. Usage: ./run.sh tools/newlevel.py <challenge> <N>
Idempotent: never overwrites existing files. Solver stub: solutions/<challenge_mod>/l<N>.py with solve(path, ctx) -> str."""
import sys
from pathlib import Path
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from cf import paths

SPEC = """# {ch} level {n} SPEC
(fill from tools/pdf2spec.py output; keep it under 40 lines)

## Task (one sentence)
## Input format (files, columns, units, missing marker)
## Output format (exact; copy the sample from the PDF)
## Metric + pass threshold per file
## Story clues that matter (quote them; story can hide mechanism / regime changes)
## Traps stated in this PDF or hinted by earlier PDFs
## Data given from earlier levels (files, what they contain)
## Open questions for the user
"""
STUB = '''"""{ch} level {n}. solve(path, ctx) -> output text for ONE input file. Reuse earlier levels via imports (snowball): from . import l1"""
from pathlib import Path
import numpy as np, pandas as pd
from cf import io as cio, metrics, models, cv, submit

def solve(path: Path, ctx) -> str:
    raise NotImplementedError("level {n}: implement after the approach is validated; see notes/{ch}/level-{n}/PLAN.md")
'''

def main():
    ch = paths.Challenge(sys.argv[1]); n = int(sys.argv[2]); lv = ch.level(n)
    nd = lv.notes_dir; nd.mkdir(parents=True, exist_ok=True)
    for name, txt in (("SPEC.md", SPEC.format(ch=ch.name, n=n)), ("PLAN.md", f"# {ch.name} level {n} PLAN\n(hypotheses, evidence, decision, status)\n")):
        if not (nd / name).exists(): (nd / name).write_text(txt)
    sd = paths.ROOT / "solutions" / ch.module; sd.mkdir(parents=True, exist_ok=True)
    (sd.parent / "__init__.py").touch(); (sd / "__init__.py").touch()
    stub = sd / f"l{n}.py"
    if not stub.exists(): stub.write_text(STUB.format(ch=ch.name, n=n))
    (paths.ROOT / "notebooks" / ch.name).mkdir(parents=True, exist_ok=True)
    print(f"scaffolded: {nd.relative_to(paths.ROOT)}/ (SPEC.md, PLAN.md), {stub.relative_to(paths.ROOT)}, notebooks/{ch.name}/L{n}.ipynb (created on first nb exec)")

if __name__ == "__main__":
    main()
