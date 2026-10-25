"""python arena.py A.onnx B.onnx games [sims] [seed0] -> A's win rate, sides swapped, referee MCTS, 2 procs"""
import sys, time, numpy as np
sys.path.insert(0, "/tmp/work/l3"); sys.path.insert(0, "/tmp/work/l3/play")
from multiprocessing import Pool
from pathlib import Path
from colony_one import constants
from model_vs_model import SearchAgent, play_game
def one(a):
    pa, pb, seed, sims, swap = a
    A = SearchAgent(Path(pa), sims, seed); B = SearchAgent(Path(pb), sims, seed + 1)
    cy, mg = (B, A) if swap else (A, B); w, n = play_game(cy, mg)
    return (w == constants.MAGENTA) if swap else (w == constants.CYAN), n
if __name__ == "__main__":
    pa, pb, g = sys.argv[1], sys.argv[2], int(sys.argv[3]); sims = int(sys.argv[4]) if len(sys.argv) > 4 else 100; s0 = int(sys.argv[5]) if len(sys.argv) > 5 else 5000
    t = time.time()
    with Pool(2) as p: r = p.map(one, [(pa, pb, s0 + i, sims, i % 2 == 1) for i in range(g)], chunksize=2)
    w = sum(x[0] for x in r); n = len(r); p_ = w / n; se = (p_ * (1 - p_) / n) ** 0.5
    print(f"A={Path(pa).name} wins {w}/{n} = {p_:.1%} (+-{1.96*se:.1%})  avg len {np.mean([x[1] for x in r]):.0f}  {time.time()-t:.0f}s")
