---
name: cf-nb
description: How to run Python cells with persistent state on the user's folder despite 180s calls and no background processes. Use whenever you explore data, train, or debug in this project.
---
# Stateful cell runner (tools/nb.py)

In Mode R there is no live kernel (processes die when a call ends, calls are capped at 180 s); in Mode L the tool is still the default because it behaves the same everywhere. `tools/nb.py` fakes a kernel: each call starts a fresh kernel, restores your variables from a checkpoint, runs ONE cell, saves variables again, records the cell and its outputs in the .ipynb.

```
./run.sh tools/nb.py exec notebooks/<ch>/L2.ipynb --challenge <ch> -c "code"      # append + run a cell
./run.sh tools/nb.py exec notebooks/<ch>/L2.ipynb --scratch -c "code"            # run, do not record the cell
./run.sh tools/nb.py exec notebooks/<ch>/L2.ipynb --replace-last -c "fixed code" # redo the last cell (state rolls back first)
./run.sh tools/nb.py exec notebooks/<ch>/L2.ipynb -f /path/snippet.py            # code from a file (long cells)
./run.sh tools/nb.py run  notebooks/<ch>/L2.ipynb                                # fresh top-to-bottom rerun (reproducibility check)
./run.sh tools/nb.py show / reset  notebooks/<ch>/L2.ipynb
```
- Kernel starts with `from cf.nbinit import *`: np, pd, plt, Path, ROOT, cio, metrics, models, cv, submit, paths, profile, disk_cache, and `ch` (the Challenge, if `--challenge` was given).
- A cell that errors or times out has NO effect: not recorded, state not saved. The error is printed.
- Variables starting with `_` are not saved (scratch). Unpicklable objects are listed in the `[state]` line; recompute them.
- Cell timeout 165 s (`--timeout` or `CF_NB_TIMEOUT`; keep it below the shell's own cap). For longer work: subsample, split into several cells, cache with `@disk_cache`, or reduce dimensions first. In Mode R do not start background jobs (they die); in Mode L a background script with polling is fine.
- Memory (Mode R: 3.9 GB): use float32/int16 (see recon notebook: 1.9M rows x 7 cols is 47 MB as int16/float32), avoid wide pivots of the full panel in float64.
- Output is truncated to 6000 chars; print summaries (`.describe()`, `.head()`, value counts), not whole frames.
- Figures: saved to `notebooks/<ch>/figs/` and the path is printed. To look at one: Mode L, Read the PNG; Mode R, `device_stage_files` that path, then Read it.
- Put reusable code in `solutions/` or `cf/`, import it; a cell that defines a 40-line function should instead be a file. Notebooks are the lab book, `solutions/` is the product.
- Quick one-off scripts without state: `./run.sh -c "..."` or `./run.sh path/to/script.py`.
