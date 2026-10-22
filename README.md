# Cloudflight AI challenge kit

A Claude-driven workflow for the Cloudflight AI coding contest (multi-level, 4 hours, each level builds on the previous). Claude parses the level PDFs, profiles the data, proposes approaches, implements them as reusable per-level modules, and writes validated output files. The human is the consultant: validates direction, uploads files, reports pass/fail. Built and dry-run against last year's "Migratory Birds" challenge.

Nothing here is contest data. Challenge folders, PDFs, per-person solutions, outputs and notebooks are git-ignored on purpose (see `.gitignore`).

## Quick start

Needs Python 3.9+, internet (pip, GitHub, HuggingFace, PyTorch weights), about 4 GB disk for the env, a Claude surface that can run shell commands in the repo.

1. Clone the repo and open a terminal in it.
2. Build the env (outside the repo, in `~/cfenv`; override with `CF_VENV`): run these one after another.
   `python setup_env.py venv`, `python setup_env.py torch`, `python setup_env.py core`, `python setup_env.py verify`
   Add `python setup_env.py extra` only for NLP, CV or RL tasks. Cold install takes about 6 minutes. On Linux/macOS/WSL `./setup_env.sh <stage>` is the same.
3. Put the contest levels in a folder named after the challenge, for example `my-challenge/level-1/` with the PDF and input files (zips are unpacked by `tools/scan.py`).
4. Start Claude in the repo root (Claude Code CLI/desktop/IDE, or a Cowork chat linked to the computer with this folder selected) and paste a kickoff prompt from `knowledge/kickoff_prompts.md`.

Claude reads `CLAUDE.md` (operating manual) and `.claude/skills/*/SKILL.md` (playbooks). `AGENTS.md` points other agents to the same files.

## Two ways to run

Mode L (local shell): Claude's shell runs on your machine in the repo. No special limits.
Mode R (remote): a cloud session linked to your computer reaches the repo only through a Linux VM (`device_bash`). Measured limits: 2 CPUs, 3.9 GB RAM, no GPU, 180 s per call, background processes die when a call ends. The kit is built to work under those limits; `CLAUDE.md` explains how Claude tells the modes apart.

## The loop

Per level: scan, parse the PDF (story kept, it hides clues), spec, profile every file, baseline bake-off, proposal to you, implement in `solutions/<challenge>/lN.py`, run (`tools/run_level.py` must reproduce the organizer's example exactly), validate format, you upload and report pass/fail, retro into `LESSONS.md`. Level 1 and other trivial levels take a fast path.

## Layout

- `CLAUDE.md`, `AGENTS.md`, `.claude/skills/`: instructions for the agent (intake, eda, solve, nb, submit, playbook, retro).
- `cf/`: library (readers for dirty contest CSVs, metrics incl. top-K overlap, model factory, CV and bake-off, auto-EDA and shift check, output validators, disk cache, notebook state).
- `tools/`: `scan.py`, `pdf2spec.py`, `newlevel.py`, `run_level.py`, `nb.py` (stateful notebook cells without a live kernel), `verify_env.py`.
- `knowledge/`: contest patterns, checklist, kickoff prompts, and a dry-run answer key for migratory-birds (its name says ANSWERS: do not read it in a blind run).
- `LESSONS.md`: running log of traps and rules. Add to it during every run and commit it.
- `solutions/`, `notes/`, `notebooks/`, `submissions/`: local working state, created per challenge, ignored by git.
- `tests/`: `python run.py -m pytest -q tests`.

## For the agent: replicating this workflow on a new machine

1. Read `CLAUDE.md` fully, then `LESSONS.md` (top), then the skill you need.
2. Detect the mode (Mode L or R, see `CLAUDE.md`).
3. If `~/cfenv` is missing, run the setup stages above (separate calls in Mode R).
4. `python run.py -m pytest -q tests` must pass; `python run.py tools/scan.py` lists the challenges found.
5. Follow the per-level loop. Never assume the output type (txt or csv): it is stated in each PDF.

## Known limits

Tested on Linux only (a Linux VM). Windows and macOS paths are handled in `setup_env.py` and `run.py` but not exercised yet: if something breaks, fix it and log it in `LESSONS.md`. OCR in `pdf2spec.py` needs the `tesseract` CLI and is optional. No GPU is assumed; torch is the CPU build.

## Maintainer notes

- After the first `git add`, mark the scripts executable if your checkout loses the bit: `git update-index --chmod=+x run.sh setup_env.sh`.
- To share a solution set on purpose: `git add -f solutions/<challenge>/`.
- Do not commit contest PDFs or data (terms of the organizer).
