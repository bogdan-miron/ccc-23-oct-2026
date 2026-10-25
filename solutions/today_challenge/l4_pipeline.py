"""Unattended loop. python pipeline.py START_ITER ITERS "data1.npz data2.npz ..." (init from models/best.pt, models/best.onnx)
Each iter i: train candidate on all data (init best) -> arena vs best (N games) -> promote if win rate >= THRESH -> generate new games with best."""
import sys, os, subprocess, shutil, re, time
os.chdir("/tmp/work/l3"); PY = sys.executable; THRESH = 0.58; ARENA = 100; GEN = 700; EPOCHS = 4
def run(cmd, log):
    with open(log, "a") as f: f.write(f"\n$ {cmd}\n"); f.flush(); return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
def say(s):
    with open("models/pipeline.log", "a") as f: f.write(time.strftime("%H:%M:%S ") + s + "\n")
start, iters, data = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3].split()
for i in range(start, start + iters):
    say(f"iter {i}: training on {data}")
    run(f"{PY} mine/train.py models/cand{i}.onnx {EPOCHS} {' '.join(data)} --init models/best.pt --lr 1e-3", f"models/cand{i}.log")
    if not os.path.exists(f"models/cand{i}.onnx"): say(f"iter {i}: training failed"); break
    v = subprocess.run(f"{PY} tools/validate_submission.py models/cand{i}.onnx", shell=True, capture_output=True, text=True).stdout
    if "ACCEPTED" not in v: say(f"iter {i}: validator rejected candidate"); break
    r = subprocess.run(f"{PY} mine/arena.py models/cand{i}.onnx models/best.onnx {ARENA} 100 {30000 + i * 1000}", shell=True, capture_output=True, text=True).stdout
    say(f"iter {i}: {r.strip()}")
    m = re.search(r"wins (\d+)/(\d+)", r); wr = int(m.group(1)) / int(m.group(2)) if m else 0
    if wr >= THRESH:
        shutil.copy(f"models/cand{i}.onnx", "models/best.onnx"); shutil.copy(f"models/cand{i}.pt", "models/best.pt")
        shutil.copy(f"models/cand{i}.onnx", f"/mnt/user-data/outputs/l4_cp_iter{i}.onnx")
        say(f"iter {i}: PROMOTED -> models/best.onnx, copy at /mnt/user-data/outputs/l4_cp_iter{i}.onnx")
    else: say(f"iter {i}: not promoted ({wr:.0%} < {THRESH:.0%})")
    run(f"{PY} mine/selfplay.py models/best.onnx {GEN} data/gen{i}.npz {40000 + i * 5000} 100 2", f"data/gen{i}.log")
    if os.path.exists(f"data/gen{i}.npz"): data.append(f"data/gen{i}.npz")
    say(f"iter {i}: generated data/gen{i}.npz, data files now {len(data)}")
