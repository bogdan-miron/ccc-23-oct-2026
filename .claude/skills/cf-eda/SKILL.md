---
name: cf-eda
description: Exploratory data analysis for a contest level. Profiles every input file, hunts data traps (dirty tokens, unit bugs, shift, join keys, hidden deterministic rules) and writes EDA.md. Use after cf-intake and whenever a model disagrees with expectations.
---
# EDA (target: 10 to 15 minutes, numbers not prose)

Run inside the level notebook: `./run.sh tools/nb.py exec notebooks/<ch>/L<N>.ipynb --challenge <ch> -c "..."` (see cf-nb).

1. Profile each input file: `profile.profile(path)`. Files a,b,c,d,e differ on purpose (size, ties, dirty values, all-missing target): profile ALL of them, not only the biggest.
2. Trap checklist, one line of evidence each:
   - dirty tokens (words as numbers, stray text): profile prints DIRTY lines; `cio.to_number` fixes and reports.
   - unit/scale bugs: bimodal histogram or a gap (L1 file e: temperature 5..45 and 55..113, degree F mixed with C). Values in the ambiguous overlap cannot be fixed by range alone: use a second signal (relationship with the target, neighbors, residuals).
   - missing marker is the prediction target; which rows are missing and why (random, blocked, whole entity)?
   - duplicates, id uniqueness, id ranges. Do the test files partition one universe (L2 a/b/c did)? If so pool them.
   - join keys with other files and earlier levels: `profile.join_keys({...})`. Check coverage before merging.
   - ordering, ids or row order that correlate with the target (leak or structure).
   - distribution shift between train and test or between regimes: `profile.compare(a, b)` (adversarial AUC above 0.7 is real, but it is degenerate when a feature is constant within a group such as wind per day: then compare per-feature KS and day-level aggregates).
   - constants and degenerate files (L1 d: all temperatures equal, so tie-break rules decide everything).
3. Hidden rule test: fit a quick GBM and a linear model with the obvious features. R2 near 1 means a deterministic rule: find it (look at residual structure, try products/ratios/thresholds). R2 stuck far below the noise implied by the threshold means a missing feature: look in other files and in the story.
4. Structure for CV: iid rows, groups (flock, entity), or time. Choose the split that mimics the test, say it out loud.
5. Write `notes/<ch>/level-N/EDA.md`: at most 30 lines, each finding as "number: so what". Add the open traps to SPEC.md.
6. Plots only when numbers fail you (a plot costs extra round trips, in Mode R a stage + Read). Save figures with plt.savefig under notebooks/<ch>/figs, then view it (Mode R: stage first).
