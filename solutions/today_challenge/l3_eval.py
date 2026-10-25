"""python eval.py model.onnx games [sims] [seed0]  -> cyan model (MCTS) vs GreedyAgent magenta"""
import sys, time, numpy as np
sys.path.insert(0, "/tmp/work/l3"); sys.path.insert(0, "/tmp/work/l3/play")
from multiprocessing import Pool
from colony_one import constants
from colony_one.baselines import GreedyAgent
from colony_one.env import ColonyOneEnv
from model_vs_model import SearchAgent, play_game
from pathlib import Path
def one(a):
    path, seed, sims = a
    cy = SearchAgent(Path(path), sims, seed); gr = GreedyAgent(seed + 1000)
    t = time.time(); w, n = play_game(cy, gr); return w, n, time.time() - t
if __name__ == "__main__":
    path, games = sys.argv[1], int(sys.argv[2]); sims = int(sys.argv[3]) if len(sys.argv) > 3 else 60; s0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    with Pool(2) as p: res = p.map(one, [(path, s0 + i, sims) for i in range(games)])
    wins = sum(r[0] == constants.CYAN for r in res)
    print(f"{path}: CYAN wins {wins}/{games}  avg actions {np.mean([r[1] for r in res]):.0f}  avg s/game {np.mean([r[2] for r in res]):.1f}  max {max(r[2] for r in res):.1f}")
