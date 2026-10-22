# Pre-contest checklist (do before the clock starts)
1. Open this chat linked to the computer (desktop app), send any message, confirm device tools answer (`get_device_info`).
2. `./setup_env.sh venv`, `torch`, `core`, `verify` (or `python setup_env.py <stage>`), separate calls in Mode R (about 40 s, 2-3 min, 1 min). `extra` only if the task is NLP/CV/RL.
3. Create `<challenge>/` in the folder, drop level files (zip or folders). `./run.sh tools/scan.py`.
4. Kickoff message: "Read CLAUDE.md. Contest is live. Challenge folder: <name>."
5. Keep the platform open in another window; upload from `submissions/<challenge>/level-N/`.
6. Tell Claude the verdict per file after each upload (pass/fail, and score if the platform shows it).

# Dry run on migratory-birds (blind mode)
Use a FRESH chat linked to the computer so CLAUDE.md and the skills are what is tested, not memory of the setup chat.
Kickoff: "Read CLAUDE.md. This is a dry run on last year's challenge. Challenge folder: migratory-birds. Do not open anything with ANSWERS or DO_NOT_READ in its name. Log lessons in LESSONS.md and minutes per level in notes/migratory-birds/timeline.md."
After the dry run, the answers files may be compared against what the pipeline found.
