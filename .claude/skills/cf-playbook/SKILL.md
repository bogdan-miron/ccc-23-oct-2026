---
name: cf-playbook
description: Model and method selection for contest tasks (tabular regression/classification, entity classification from sequences, panel forecasting and top-K ranking, regime shift, NLP, CV, RL/optimization). Use when choosing an approach or when a baseline misses the threshold.
---
# Playbook: first move, strong baseline, CV, trap, intended trick

General: the data is synthetic, so the generating rule exists. Order of effort: (1) understand 10 rows, (2) baseline, (3) residuals, (4) features from story and constraints, (5) rule search, (6) model upgrade last. No tuning marathons. No stacking before one model is understood.

## Tabular regression with missing targets (L2 type)
- First: pool all files, join extra features from earlier levels (check key coverage), fix unit bugs, `cv.bakeoff`.
- If all models plateau at the same RMSE: noise floor or missing feature. Missing feature is the usual answer: look at other files, story, interactions (ratios, products, thresholds). Fit a GBM on all features, then inspect residual vs each feature.
- Unit-bug rows (degree F in a C column): confident cases by range (values above any plausible C). Ambiguous overlap: build both interpretations, train on confident rows, classify ambiguous rows by residual, or give the model `raw`, `fixed_if_gt_threshold`, `is_ambiguous` features. Verify with CV on rows with known truth.
- Predicting for the file that is 100% unlabeled (like c): train on all labeled rows of all files.

## Entity classification from sequences (L3/L4 type)
- The story describes each class's behavior in words: turn each sentence into a feature and test it with a `groupby(class).mean()` table (separability first, models second). Typical: palindromic route, returns to start, revisits a hub, split return path, orbit around a center, overlap of segments with another class's routes, mean temperature of visited nodes (needs the clean node attributes).
- Aggregate rows to the entity (mean/std/min/max of row features, counts of identical rows, shared prefixes) then classify the entity. CV with GroupKFold by entity.
- Constraints: "each class appears exactly once per dataset" is an assignment problem: score matrix [entities x classes] then `scipy.optimize.linear_sum_assignment`. Report both argmax and assignment results.
- Metric F1 macro: check per-class confusion, not only the mean.

## Panel forecasting and top-K ranking (L5 type)
- Backtest like the test: hide the last K days, forecast, score with `metrics.topk_overlap`. Baselines first (trailing mean 7/30, last value, EWMA). Last year persistence scored 0.57 vs a 0.5 threshold.
- Check which columns exist for forecast rows (covariates given? occupancy given?). Use only what is available at forecast time. Wind or weather that is global per day is a time series, not a per-entity feature.
- Arrays beat DataFrames: pivot to [days, entities] float32, vectorize with numpy.
- Graph hints ("reuse path data from level 4"): build adjacency from paths, use neighbor aggregates of occupancy/arrivals.
- Rank metric rewards the set: predict expected value, take top K; consider averaging ranks over horizons.

## Regime shift / twist (L6 type)
- Detect: `profile.compare(old, new)`, adversarial AUC, per-feature KS. Read the story: "wind blows the opposite direction" means sign flip, "moving backwards" means reversed order or sign.
- Candidates: (a) map new covariates back to the old regime and reuse the big model, (b) refit on the new regime only, (c) hybrid with a regime flag. Validate on labeled new-regime days with a time split.
- 100% required means a deterministic rule: find the rule that reproduces the new labeled days exactly, then extrapolate it one step.

## Text
- Look for templates first (regex, token counts): synthetic text is often template-generated. Then TF-IDF (word + char n-grams) + logistic regression / linear SVM. Then sentence embeddings (`SentenceTransformer("all-MiniLM-L6-v2")`, HuggingFace works from the VM) + logistic regression. Fine-tuning a transformer on CPU only for a few thousand short texts.

## Images
- Inspect size/channels/classes; for synthetic shapes use classic features (color histograms, contours with cv2). Pretrained `timm`/torchvision resnet18 features + logistic regression; precompute embeddings once and cache them (CPU, 2 cores: budget minutes per few thousand images).

## RL, simulation, optimization, graphs
- Reimplement the environment exactly from the spec, unit-test it on the example trajectory. Heuristic/greedy baseline, then search (beam, A*, DP), then CP-SAT (`ortools`) for assignment/scheduling. RL only if the environment is small and heuristics fail. Fix seeds.
- `networkx` for shortest paths and components.

## Anti-patterns
Tuning for hours; scaling features for trees; using test-fold statistics (leak); ignoring that small files probe edge cases; adding model complexity to reach a threshold a rule would hit; coding before seeing 10 rows; trusting a local score from a split that does not mimic the test; plots instead of numbers.
