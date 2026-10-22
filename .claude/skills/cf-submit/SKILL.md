---
name: cf-submit
description: Produce and validate the files the user uploads to the contest platform, and handle the pass/fail loop. Use whenever outputs are generated or the user reports an upload result.
---
# Outputs and upload loop

0. Output type is set by this year's PDF (txt or csv, extension, one file per input or one combined file). Read it from the acceptance page and the sample; do not carry over last year's choice.
1. Produce: `./run.sh tools/run_level.py <ch> N` writes `submissions/<ch>/level-N/out_level-N_K-name.<ext>` for every `in_` file. Name rule: `in_` becomes `out_`, same stem and extension as the input unless the PDF says otherwise (last year: txt for L1, csv content in txt files for the rest; confirm from the PDF and example).
2. Hard checks (run_level does them; read the result, do not assume):
   - example output equals the organizer's `out_..._0-example` (whitespace and CRLF tolerant);
   - header string identical to the sample (units, brackets, case);
   - every id that was `missing` appears exactly once, no extras (`submit.check_output(expect_ids=...)`);
   - no nan/None/inf, no BOM, field counts match the sample, row count matches expectation.
3. Style: organizer outputs use CRLF and sometimes no trailing newline. `submit.write_text(path, text, like=example)` mimics the example. The checker normalizes, so this is cosmetic but cheap.
4. Tell the user: the folder `submissions/<ch>/level-N/` (give the absolute path on his machine if you know it), the exact files to upload, local score vs threshold per file, the risky file(s), and the fallback you will try.
5. Pass/fail intake: ask for the verdict per file. Record it in `notes/<ch>/level-N/PLAN.md` (date, variant, verdict). Pass: `run_level.py <ch> N --freeze`. Fail: cf-solve phase E.
6. Variants: when two hypotheses differ, put variant A in some files and variant B in others where the platform allows per-file verdicts, so one upload round answers both. Keep variants under `variants/`.
7. Regression guard: after changing shared code, `run_level.py <ch> N --regress` re-runs earlier levels and diffs with their frozen passed outputs.
