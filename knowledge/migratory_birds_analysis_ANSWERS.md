# migratory-birds: level by level (facts verified from the files; hypotheses marked H, to be tested in the dry run)

## L1 sort BOPs (txt, space separated list)
- Facts: sort key = temperature desc, humidity asc, BOP asc (matches the example). Files: a 5 rows (has "seventeen"), b 51 (sparse ids), c 200, d 73 (all temperatures 19: tie-break test), e 2171 (759 values 55..113 = degree F, plus "seventeen").
- H: expected answers for e use corrected temperatures. Test with variants on the platform (raw vs F-fixed for values > 45). Values 32..45 that are F are indistinguishable from C by range.

## L2 predict Bird Love Score (csv, header `BOP,Bird Love Score [<3]`)
- Facts: a/b/c = 1000/1400/100 rows, disjoint, partition BOP 1..2500; labels missing 20% / 14% / 100%. Join temperature and humidity from `all_data_from_level_1.in`. Thresholds RMSE < 6 / 3 / 3.
- Baseline (5-fold CV on 2000 labeled rows, F fixed for t > 45): ridge 2.58, rf 2.79, lgbm 2.87. Without F fix ridge 13.1, rf 2.81. Temperature dominates the linear fit (std coef 21 vs 0.5 for the next feature).
- H: a linear (or smooth monotone) rule in temperature plus small terms; remaining noise about 2.4 after removing 2% outliers (16 of 44 big residuals have raw t > 60, check F conversion rounding).

## L3 flock -> species (6 flocks per file, each species exactly once)
- Facts: files a/b/c have 6 flocks and 25-31 paths. Species behaviors (from the story) listed in notes/migratory-birds/level-3/pdf.md.
- H: features per flock: palindromic share, first==last, hub revisits, identical rows, mean temperature of visited BOPs (F-fixed), overlap with the Bluetit (palindromic) routes. Solve with scores + Hungarian assignment.

## L4 same at scale (csv, `Flock ID,Species`, F1 macro > 90%)
- Facts: 360 flocks, 288 labeled (all rows of a flock share the label) and 72 unlabeled whole flocks, 3-12 rows per flock, ~45-50 flocks per class, 90 duplicate rows.
- H: L3 features + GroupKFold by flock, classify per flock, broadcast.

## L5 top-50 arrival BOPs per day, days 731-760 (csv `Day,Top 50 Arrivals BOPs`, average accuracy >= 50%)
- Facts: 1.9M rows = 760 days x 2500 BOPs; Arrivals missing for days 731-760; Occupancy, wind and insects given for those days. Wind is one value per day, Insects Delta is constant per BOP. Autocorrelation: arrivals 0.96, occupancy 0.996, wind 0.92. Backtest (hide days 701-730): trailing-30-day mean per BOP top-50 overlap 0.57, trailing 7 0.55, last day 0.54, mean of all history 0.40; day-to-day top-50 turnover 0.86.
- H: path data from L4 defines a BOP graph; arrivals depend on neighbors' occupancy and wind. A model with future covariates should beat persistence.

## L6 top-50 for day 791 (accuracy 100%)
- Facts: days 761-790 given with Arrivals; no covariates for day 791. Per-BOP insects identical to L5. Wind changed: wx mean -0.11 -> -0.25, wy mean -0.09 -> +0.03, wy sd 0.33 -> 0.17. Arrivals mean 112.6 -> 102.4.
- H: the story says wind flips and "worms move backwards": the arrival mechanism in the new regime may be the old one with transformed covariates. Find the rule that reproduces days 761-790 exactly, then roll it one day.
- Tool note: adversarial AUC is degenerate here (wind is constant per day, so any day is identifiable). Compare day-level aggregates or use KS per feature.
