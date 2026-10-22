---
name: cf-intake
description: Start of a new contest level or a new challenge folder. Scans the folder, parses the decorated PDF into clean text, scaffolds the level and writes SPEC.md. Use when new level files appear or the user says a level is ready.
---
# Level intake (target: under 5 minutes, brief to the user at the end)

1. Inventory: `./run.sh tools/scan.py [challenge]` (unzips new zips, lists levels, inputs, examples, extras, marks NEW).
2. Parse the PDF: `./run.sh tools/pdf2spec.py <challenge>/level-N/level-N.pdf -o notes/<challenge>/level-N/pdf.md --pages` and read the whole output (about 1-2 screens).
   - Boilerplate is removed, story text is KEPT on purpose: it hides clues.
   - If it warns about image-only pages, view them: Read the PNG in `notes/.../pages/` (Mode R: `device_stage_files` it first), or rerun with `--ocr` (needs the tesseract CLI).
3. Scaffold: `./run.sh tools/newlevel.py <challenge> N`, then fill `notes/<challenge>/level-N/SPEC.md` (under 40 lines): task, input format, exact output format, metric + threshold per file, story clues, traps, data handed over from earlier levels, open questions.
4. Understand the task on the tiny example by hand: take `in_..._0-example` and `out_..._0-example`, reproduce the output mentally or with 3 lines of code. If you cannot, the spec is wrong.
5. Scan for clue words in the story and bullets: opposite, reversed, changed, mistakenly, bug, only, exactly, each, all, never, unit names, "reuse data from level k". Quote them in SPEC.md.
6. Cross-check against earlier levels: does this PDF reveal a trap in data you already used (unit bug, label noise)? If yes: log to LESSONS.md, re-run the earlier solver with the fix, run `tools/run_level.py <ch> N --regress`, and tell the user which earlier outputs changed.
7. Output format details: file extension and name come from the example (`in_` becomes `out_`). Check CRLF/trailing newline handling is mimicked (cf.submit does it). Confirm the header text of the sample output character by character (units and brackets included).
8. Brief to the user (max 10 lines): task in one sentence, files, metric and threshold, traps spotted, first idea, what you will run next. Then continue with cf-eda without waiting unless a question changes the plan.
