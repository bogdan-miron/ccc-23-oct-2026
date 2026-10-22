---
name: cf-retro
description: After each level (and at the end of the contest or a dry run): log lessons, update knowledge and tools, record timings. Use right after a pass/fail verdict, and whenever a mistake or time sink happens.
---
# Retro (2 minutes per level, do not skip)

1. Append to `LESSONS.md` under the right section: `ID (level, date) what happened | cost in minutes | rule going forward`. Facts and numbers only. Include things that wasted time and things that saved it.
2. Update `knowledge/cloudflight_patterns.md` when a new pattern or trap type shows up.
3. Promote ad-hoc code that was used twice into `cf/` (with a one-line docstring) or into `solutions/`. If a tool failed or was missing, fix the tool now, not "later".
4. If the workflow itself was wrong (a skill step was useless, a rule too slow, a question that should have been asked earlier), edit the skill or CLAUDE.md and note it in LESSONS.md under Process.
5. Timeline: append to `notes/<challenge>/timeline.md`: level, start, first output, verdict, minutes.
6. At the end: `notes/<challenge>/RETRO.md` (what to keep, what to change, time per level, biggest misses).
