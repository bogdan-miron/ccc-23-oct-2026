# Level 4 results (local, 100 sims, sides swapped)
- hand model: baseline. cp1 beat hand 54/60. cp2 beat cp1 80/100. cand3 vs cp2 53/100 (not promoted).
- cp1 beat v3 65/80. cp2 vs v7 50/100, vs v8 49/100. ens(cp2+v8) vs cp2 55/100.
- v9 beat cp2 61/100. ens(cp2+v9) vs v9 41/100. v10 vs v9 40/80. v10 beat cp2 71/100.
- Recommendation: upload v10.
Notes: model self-play ~2 s/game/core; training OOM fixed by float16 targets (train.py).
