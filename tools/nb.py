"""Stateful notebook cell runner (no live kernel possible: processes die after each call, calls are capped at 180s).
Every exec: fresh kernel -> restore variables -> run ONE cell -> checkpoint variables -> record cell+outputs in the .ipynb.
Usage (via ./run.sh tools/nb.py ...):
  exec  NB.ipynb -c "code" | -f file.py   [--challenge NAME] [--scratch] [--replace-last] [--timeout 165] [--max-chars 6000]
  run   NB.ipynb [--upto J]                 re-run all cells top to bottom in a fresh kernel (reproducibility check) and save outputs
  show  NB.ipynb                            list cells
  reset NB.ipynb                            drop saved variables (keeps cells)
A failed or timed-out cell has NO effect (not recorded, state not saved). Variables starting with '_' are not persisted."""
import argparse, base64, os, re, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nbformat
from nbformat import v4 as nbf
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError, DeadKernelError

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get("CF_STATE", Path.home() / ".cfstate")) / "nb"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

def sdir(nb_path, which="cur"):
    key = re.sub(r"\W+", "_", str(Path(nb_path).resolve().relative_to(ROOT)) if Path(nb_path).resolve().is_relative_to(ROOT) else Path(nb_path).stem)
    return STATE / key / which

def load_nb(p):
    p = Path(p)
    if p.exists(): return nbformat.read(p, 4)
    nb = nbf.new_notebook(); nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}; return nb

def run_cells(cells, timeout, restore_dir=None, save_dir=None, challenge=None, stop_on_error=True):
    os.environ.pop("MPLBACKEND", None); os.environ["CF_ROOT"] = str(ROOT); os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if challenge: os.environ["CF_CHALLENGE"] = challenge
    init = nbf.new_code_cell("from cf.nbinit import *\nfrom cf import state as _st\n" + (f"_st.restore({str(restore_dir)!r}, globals())" if restore_dir else ""))
    post = nbf.new_code_cell(f"_r = _st.checkpoint({str(save_dir)!r}, globals())") if save_dir else None
    nb = nbf.new_notebook(cells=[init, *cells] + ([post] if post else []))
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True, record_timing=False, resources={"metadata": {"path": str(ROOT)}})
    err = None; t0 = time.time(); times = []
    with client.setup_kernel():
        for i, c in enumerate(nb.cells):
            if c is post and err: break
            ts = time.time()
            try:
                client.execute_cell(c, i)
            except CellTimeoutError:
                err = f"TIMEOUT after {timeout}s in cell {i} (calls are capped at 180s: subsample, cache, or run in the cloud container)"; break
            except DeadKernelError:
                err = "KERNEL DIED (probably out of memory: 3.9GB RAM; use float32/polars/chunks)"; break
            times.append(time.time() - ts)
            if any(o.output_type == "error" for o in c.outputs): err = err or f"error in cell {i}"; 
            if err and stop_on_error: break
    return nb, err, times

def show_outputs(cell, figdir, tag, max_chars):
    out, k = [], 0
    for o in cell.outputs:
        if o.output_type == "stream": out.append(o.text)
        elif o.output_type in ("execute_result", "display_data"):
            d = o.data
            if "image/png" in d:
                figdir.mkdir(parents=True, exist_ok=True); f = figdir / f"{tag}_{k}.png"; k += 1; f.write_bytes(base64.b64decode(d["image/png"])); out.append(f"[image saved: {f.relative_to(ROOT)}]\n")
            if "text/plain" in d and not ("image/png" in d and d["text/plain"].startswith("<Figure")): out.append(d["text/plain"] + "\n")
        elif o.output_type == "error": out.append(ANSI.sub("", "\n".join(o.traceback[-6:])) + "\n")
    s = "".join(out)
    return s if len(s) <= max_chars else s[:max_chars // 2] + f"\n...[truncated {len(s)-max_chars} chars; full output in notebook]...\n" + s[-max_chars // 2:]

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exec"); e.add_argument("nb"); e.add_argument("-c"); e.add_argument("-f"); e.add_argument("--challenge"); e.add_argument("--scratch", action="store_true")
    e.add_argument("--replace-last", action="store_true"); e.add_argument("--timeout", type=int, default=int(os.environ.get("CF_NB_TIMEOUT", 165))); e.add_argument("--max-chars", type=int, default=6000)
    r = sub.add_parser("run"); r.add_argument("nb"); r.add_argument("--upto", type=int); r.add_argument("--challenge"); r.add_argument("--timeout", type=int, default=int(os.environ.get("CF_NB_TIMEOUT", 165))); r.add_argument("--max-chars", type=int, default=3000)
    s = sub.add_parser("show"); s.add_argument("nb")
    z = sub.add_parser("reset"); z.add_argument("nb")
    a = ap.parse_args(); nbp = Path(a.nb)
    if not nbp.is_absolute(): nbp = ROOT / nbp
    figdir = nbp.parent / "figs"
    if a.cmd == "show":
        nb = load_nb(nbp)
        for i, c in enumerate(nb.cells): print(f"[{i}] {c.cell_type}: {c.source.strip().splitlines()[0][:90] if c.source.strip() else ''}")
        return
    if a.cmd == "reset":
        for w in ("cur", "prev"): shutil.rmtree(sdir(nbp, w), ignore_errors=True)
        print("state cleared"); return
    nb = load_nb(nbp)
    if a.cmd == "run":
        cells = [c for c in nb.cells if c.cell_type == "code"][: (a.upto + 1) if a.upto is not None else None]
        for c in cells: c.outputs = []; c.execution_count = None
        shutil.rmtree(sdir(nbp, "prev"), ignore_errors=True); shutil.rmtree(sdir(nbp, "cur"), ignore_errors=True)
        ran, err, times = run_cells(cells, a.timeout, None, sdir(nbp, "cur"), a.challenge)
        for c, rc in zip(cells, ran.cells[1:]): c.outputs, c.execution_count = rc.outputs, rc.execution_count
        nbformat.write(nb, nbp)
        for i, (c, t) in enumerate(zip(cells, times[1:])): print(f"[cell {i}] {t:.1f}s {'ERROR' if any(o.output_type=='error' for o in c.outputs) else 'ok'}")
        if err: print("STOPPED:", err); print(show_outputs(ran.cells[len(times) - 1], figdir, nbp.stem, a.max_chars)); sys.exit(1)
        print(f"[run] {len(cells)} cells ok, total {sum(times):.0f}s"); return
    code = a.c if a.c is not None else (Path(a.f).read_text() if a.f else sys.stdin.read())
    cell = nbf.new_code_cell(code.rstrip("\n"))
    cur, prev = sdir(nbp, "cur"), sdir(nbp, "prev")
    restore_from = prev if a.replace_last else cur
    tmp = cur.parent / "tmp"; shutil.rmtree(tmp, ignore_errors=True)
    ran, err, times = run_cells([cell], a.timeout, restore_from if restore_from.exists() else None, tmp, a.challenge)
    c = ran.cells[1]
    if any(o.output_type == "error" for o in ran.cells[0].outputs): print("INIT CELL ERROR:\n" + show_outputs(ran.cells[0], figdir, "init", 3000)); sys.exit(2)
    idx = len([x for x in nb.cells]) - (1 if a.replace_last and nb.cells else 0)
    print(show_outputs(c, figdir, f"{nbp.stem}_c{idx}", a.max_chars), end="")
    if len(ran.cells) > 2 and not err:
        print(show_outputs(ran.cells[2], figdir, "state", 1500), end="")
    if err:
        shutil.rmtree(tmp, ignore_errors=True); print(f"\n[nb] FAILED, nothing recorded or saved. {err if 'cell 1' not in err else ''}"); sys.exit(1)
    print(f"[nb] cell {idx} ok in {times[1]:.1f}s" + (" (scratch: not recorded)" if a.scratch else ""))
    if not a.scratch:
        if a.replace_last and nb.cells: nb.cells[-1] = c
        else: nb.cells.append(c)
        nbp.parent.mkdir(parents=True, exist_ok=True); nbformat.write(nb, nbp)
    if not a.replace_last:
        shutil.rmtree(prev, ignore_errors=True)
        if cur.exists(): shutil.move(str(cur), str(prev))
    else: shutil.rmtree(cur, ignore_errors=True)
    shutil.move(str(tmp), str(cur))

if __name__ == "__main__":
    main()
