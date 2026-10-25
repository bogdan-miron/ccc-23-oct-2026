"""python selfplay.py model.onnx games out.npz [seed0] [sims] [workers] -> self-play data (both sides use the model)
Saves obs (N,16,8,8) f16, pi (N,4096) f16 visit distribution, z (N,) f32 outcome for the side to move, moves (N,)."""
import sys, os, time, numpy as np
sys.path.insert(0, "/tmp/work/l3")
from multiprocessing import Pool
from colony_one import mcts, encoding, constants
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator
def game(a):
    path, seed, sims = a
    ev = OnnxEvaluator(path); rng = np.random.default_rng(seed); cfg = mcts.MCTSConfig(simulations=sims)
    OPP = "/tmp/work/l3/models/opp.onnx"
    ev2 = OnnxEvaluator(OPP) if (os.path.exists(OPP) and seed % 2 == 1) else None   # odd games: best vs opponent model
    opp_team = constants.CYAN if (seed // 2) % 2 == 0 else constants.MAGENTA
    env = ColonyOneEnv(seed=seed); obs, pis, teams = [], [], []; n = 0
    while not env.terminated and n < 200:
        team, al, rn = env.current_team, env.actions_left, env.round_number
        root = mcts.run(env.board, team, al, rn, (ev2 if (ev2 is not None and team == opp_team) else ev), cfg, rng=rng)
        pi = mcts.policy_target(root)
        T = 1.0 if n < 10 else 0.3
        idx = mcts.select_action_index(root, T, rng)
        obs.append(env.observation().astype(np.float16)); pis.append(pi.astype(np.float16)); teams.append(team)
        env.step(encoding.decode_action(idx, team, env.board)); n += 1
    w = env.winner
    z = np.array([1.0 if t == w else -1.0 for t in teams], np.float32)
    return np.stack(obs), np.stack(pis), z, w, n
if __name__ == "__main__":
    path, games, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    s0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0; sims = int(sys.argv[5]) if len(sys.argv) > 5 else 100; wk = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    t = time.time()
    with Pool(wk) as p: res = p.map(game, [(path, s0 + i, sims) for i in range(games)], chunksize=2)
    O = np.concatenate([r[0] for r in res]); P = np.concatenate([r[1] for r in res]); Z = np.concatenate([r[2] for r in res])
    np.savez_compressed(out, obs=O, pi=P, z=Z)
    cy = sum(r[3] == constants.CYAN for r in res)
    print(f"{games} games, {len(Z)} positions, cyan wins {cy}, avg len {np.mean([r[4] for r in res]):.1f}, {time.time()-t:.0f}s")
