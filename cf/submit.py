"""Write + validate output files. Contest example outputs use CRLF and inconsistent trailing newlines, so: mimic the example's style when it exists,
and ALWAYS run check_output before telling the user to upload."""
import re
from pathlib import Path
import pandas as pd

def style_of(path):
    b = Path(path).read_bytes(); return {"crlf": b"\r\n" in b, "trailing": b.endswith(b"\n")}

def write_text(path, text, like=None):
    """text uses \\n; converts to the style of `like` (an example out file) if given, else LF without trailing newline."""
    st = style_of(like) if like and Path(like).exists() else {"crlf": False, "trailing": False}
    t = text.replace("\r\n", "\n").rstrip("\n"); t = t + "\n" if st["trailing"] else t
    if st["crlf"]: t = t.replace("\n", "\r\n")
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(t.encode("utf-8")); return p

def csv_text(df, header=None, float_fmt=None):
    """DataFrame -> csv text; header = exact header string from the sample (units/brackets included!)."""
    d = df.copy()
    if float_fmt:
        for c in d.select_dtypes("float").columns: d[c] = d[c].map(lambda v: float_fmt % v)
    lines = [header] if header else [",".join(map(str, d.columns))]
    lines += [",".join(map(str, r)) for r in d.itertuples(index=False)]
    return "\n".join(lines)

def _norm(t): return [ln.strip() for ln in t.replace("\r\n", "\n").strip().split("\n")]

def diff_example(out, expected):
    """Exact (whitespace/newline-tolerant) comparison. Returns (ok, message)."""
    a, b = _norm(Path(out).read_text(encoding="utf-8")), _norm(Path(expected).read_text(encoding="utf-8"))
    if a == b: return True, "matches example exactly"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return False, f"line {i+1} differs:\n  got      {x[:120]}\n  expected {y[:120]}"
    return False, f"line count differs: got {len(a)}, expected {len(b)}"

def check_output(out, example_out=None, input_path=None, expect_rows=None, expect_ids=None, id_col=0, sep=","):
    """Structural validation. Returns list of problems ([] = ok). Checks: non-empty, no nan/None, header == example header,
    field counts, row count, id coverage (every id in expect_ids present exactly once, no extras)."""
    probs = []; p = Path(out)
    if not p.exists(): return [f"missing file {p}"]
    txt = p.read_text(encoding="utf-8"); lines = [l for l in _norm(txt) if l != ""]
    if not lines: return ["empty file"]
    if re.search(r"\b(nan|None|NaN|inf)\b", txt): probs.append("contains nan/None/inf")
    if "﻿" in txt: probs.append("BOM present")
    if example_out and Path(example_out).exists():
        ex = [l for l in _norm(Path(example_out).read_text(encoding="utf-8")) if l]
        if ex and sep in ex[0] and lines[0] != ex[0] and not re.match(r"^[\d\s.\-]+$", ex[0]): probs.append(f"header differs: got '{lines[0]}' expected '{ex[0]}'")
        if ex:
            nf = {len(l.split(sep)) for l in ex}
            if len(nf) == 1 and any(len(l.split(sep)) not in nf for l in lines): probs.append(f"field count differs from example ({nf})")
    if expect_rows is not None and len(lines) - (1 if sep in lines[0] and not lines[0][0].isdigit() else 0) != expect_rows:
        probs.append(f"row count {len(lines)} != expected {expect_rows} (+header)")
    if expect_ids is not None:
        body = lines[1:] if not lines[0].split(sep)[id_col].strip().lstrip("-").isdigit() else lines
        ids = [l.split(sep)[id_col].strip() for l in body]; exp = [str(i) for i in expect_ids]
        miss, extra, dup = set(exp) - set(ids), set(ids) - set(exp), len(ids) - len(set(ids))
        if miss: probs.append(f"{len(miss)} expected ids missing, e.g. {sorted(miss)[:5]}")
        if extra: probs.append(f"{len(extra)} unexpected ids, e.g. {sorted(extra)[:5]}")
        if dup: probs.append(f"{dup} duplicated ids")
    return probs
