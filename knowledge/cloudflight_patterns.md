# Cloudflight CCC Data & AI: what a contest looks like (source: migratory-birds, 6 levels)

## Structure
- Level N folder: `level-N.pdf` (slide deck, story + acceptance criteria), `in_level-N_K-name.txt`, `out_level-N_0-example.txt` (only the K=0 example), extras `all_data_from_level_(N-1).in`, `level_N_species.in` (label list).
- Levels 1-4: several input files per level (example, a, b, c, d, e). Each is graded separately, and the small ones probe edge cases (ties, sparse ids, all labels missing, dirty tokens). Levels 5-6: one big `main` file and a single output file.
- The PDF acceptance page gives: output format with a sample, metric and thresholds per file (RMSE < 6 / < 3 / < 3, F1 macro > 90%, average top-50 accuracy >= 50%, then 100%).
- Data handed down: `all_data_from_level_k.in` = union of that level's inputs with number-words fixed; other bugs remain.

## Escalation seen
1. Deterministic sort with tie-breaks (sort by temperature desc, humidity asc, id asc). Traps: number words, degree F.
2. Regression with missing targets: join level-1 data, fix F, pool files. Thresholds are tight.
3. Puzzle classification of 6 flocks into 6 species from path data. The story describes each species' movement: palindromic route, orbit around a land, split return, home base with excursions, predator overlapping the routes of another species, one that likes hot regions (needs clean temperature). "Exactly one flock per species" is an assignment constraint.
4. Same classification at scale (360 flocks, 20% unlabeled, F1 macro > 90%): the level-3 features generalize.
5. Forecast + top-50 ranking on a 2500 x 730 panel (1.9M rows). Persistence baselines are strong.
6. Twist: regime change announced in the story (wind opposite, worms backwards), only 30 days of new data, 100% accuracy required for one day.

## Recurring ideas
- Level 1 each year is basic (filtering, sums, sort with tie-breaks). Do it fast, but data traps still appear there.
- The output type (txt vs csv) is specified per contest in the PDF; it can differ from year to year.
- The story is a feature spec and a trap announcement. Clue words: opposite, backwards, mistakenly, exactly, each, only.
- Later PDFs reveal bugs in earlier data. Re-check earlier outputs when that happens.
- Data is synthetic. Thresholds that look too tight for noisy data mean a deterministic rule exists.
- Pool across files, across levels. Test files are slices of one universe.
- Easy levels reward speed. Hard levels reward understanding the generator, not tuning.
- Outputs: CRLF text, header with units exactly as in the sample, ids as in the input.

## Added after the migratory-birds dry run (2026-10-23)
These are last year's patterns. This year's challenge may differ: test them, do not assume them.
- L2 regression: the data trap (degree F in the earlier level's temperature) mattered more than the model; the label was linear in corrected temperature plus noise (floor RMSE 2.58).
- L3/L4 species: encode each story sentence as a feature (palindrome, shared prefix, disjoint middles, hot BOPs after the F fix); L3 assignment with Hungarian, L4 ExtraTrees on flock-level features, CV at flock level (0.9966).
- L5 forecast: model within-day centered log Arrivals; forecast-day Occupancy (given in the file) is the main feature; backtest on many windows, expect window variance (sd 0.09).
- L6 regime shift: test exact time-mirror/negation of the last block first; here day 791 = day 730.
