---
name: cf-solve
description: Core loop for solving a contest level: fast baseline, proposal to the user, validation, implementation in solutions/, diagnosis after a failed upload. Use for every level after EDA.
---
# Solve loop

## Fast path (L1 and any trivial deterministic level)
No bake-off, no proposal round. Read SPEC, write the rule in `solutions/`, run `run_level.py`, check the example, profile all files for traps (all of them, a..e), list upload files. One message to the user at the end with the open trap question, if any (e.g. raw vs corrected values). Target 10 minutes.

## Phase A: baseline (max 10 min)
- Tabular: `cv.bakeoff(X, y, task, metric, kind=...)` gives the default-model score board in about a minute. Compare with the pass threshold. If a default model already clears it with margin, stop: produce outputs, time matters.
- Ranking / forecasting: build a backtest that mimics the test (last K days as validation, metric = real contest metric, e.g. `metrics.topk_overlap`). Baselines: trailing mean, last value, per-entity mean.
- Classification of entities (flocks, users): aggregate rows to the entity first, classify the entity, broadcast the label.

## Phase B: gap analysis (only if the baseline misses)
- Where is the error concentrated (file, range, entity type, rows with a data trap)? Residual tables beat plots.
- Feature ideas come from the story and the stated constraints (cf-playbook). Constraints stated in the PDF are free information: encode them (assignment with `scipy.optimize.linear_sum_assignment`, uniqueness, monotonicity, group consistency).
- Search for the generating rule before reaching for bigger models.
- Hard stop on tuning: at most 3 configurations per model family, 5 minutes total.

## Phase C: proposal to the user (max 10 lines)
Finding (numbers) / Hypothesis / Test and cost / Options A,B / Recommendation / Kill criterion. Use AskUserQuestion for choices. Before sending, try to break your own idea with the cheapest test and include the result. When the user counters, run the cheapest falsifying experiment and report the number.

## Phase D: implement
- Code in `solutions/<ch_mod>/lN.py` (`solve(path, ctx) -> str`). Import earlier levels (`from . import l1`). Parametrize instead of copying. Heavy steps behind `@disk_cache`.
- `./run.sh tools/run_level.py <ch> N` : example must match exactly, format check must pass. Then sanity-check the predictions (range vs training range, class balance, duplicates, count of rows).
- Tell the user: files to upload (paths), local CV score vs threshold per file, risk notes, what you will do if one fails.

## Phase E: after the user's verdict
- Pass: `run_level.py <ch> N --freeze`, then cf-retro.
- Fail: do NOT try a random new model. Diagnose in this order: (1) which files failed (the small a/b/c files probe edge cases), (2) per-file edge cases: ties, sparse ids, dirty tokens, all-missing, size, (3) re-read the PDF text for a missed constraint, (4) is local CV mimicking the test split, (5) trap in data from an earlier level. State the new hypothesis with a number, then proceed.
- Never resubmit identical outputs. Keep variants in `submissions/<ch>/level-N/variants/` when you ask the user to test two hypotheses (cheap on levels with several files: put the variant only in one file).

## Time boxes (4h total, adapt to remaining time)
L1 10 min, L2 25, L3 25, L4 25, L5 45, L6 40, buffer 20. If a box is exceeded by 50%, tell the user and offer the best current output or a cheaper path.
