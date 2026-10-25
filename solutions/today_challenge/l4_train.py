"""python train.py out.onnx epochs data1.npz [data2.npz ...] [--init prev.pt] [--c 64 --nb 5]"""
import sys, time, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/tmp/work/l3/mine")
from net import Net, export, d4_perms, apply_sp
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("epochs", type=int); ap.add_argument("data", nargs="+")
ap.add_argument("--init"); ap.add_argument("--c", type=int, default=64); ap.add_argument("--nb", type=int, default=5)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--bs", type=int, default=256); a = ap.parse_args()
torch.set_num_threads(2)
D = [np.load(f) for f in a.data]
O = torch.from_numpy(np.concatenate([d["obs"] for d in D]).astype(np.float32)); P = torch.from_numpy(np.concatenate([d["pi"] for d in D]).astype(np.float16))
Z = torch.from_numpy(np.concatenate([d["z"] for d in D])); N = len(Z); print("positions", N, "z mean", Z.mean().item())
perms = d4_perms(); invs = [torch.from_numpy(p[2]) for p in perms]
net = Net(a.c, a.nb)
if a.init: net.load_state_dict(torch.load(a.init))
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
steps = a.epochs * (N // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
perm = torch.randperm(N); nval = min(5000, N // 20); vi, ti = perm[:nval], perm[nval:]
def batch(idx, aug):
    x, p, z = O[idx], P[idx].float(), Z[idx]
    if aug:
        k = np.random.randint(8); kk, fl, _ = perms[k]; x = apply_sp(x, kk, fl); p = p[:, invs[k]]
    return x, p, z
t0 = time.time(); step = 0
for ep in range(a.epochs):
    net.train(); sh = ti[torch.randperm(len(ti))]; tl = [0, 0, 0]
    for i in range(0, len(sh) - a.bs + 1, a.bs):
        if step >= steps: break
        x, p, z = batch(sh[i:i + a.bs], True); lp, v = net(x)
        pl = -(p * F.log_softmax(lp, 1)).sum(1).mean(); vl = F.mse_loss(v, z); loss = pl + vl
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1; tl[0] += pl.item(); tl[1] += vl.item(); tl[2] += 1
    net.eval()
    with torch.no_grad():
        x, p, z = batch(vi, False); lp, v = net(x)
        vpl = -(p * F.log_softmax(lp, 1)).sum(1).mean().item(); vvl = F.mse_loss(v, z).item(); acc = ((v > 0) == (z > 0)).float().mean().item()
    print(f"ep {ep} train pol {tl[0]/max(tl[2],1):.3f} val {tl[1]/max(tl[2],1):.3f} | held-out pol {vpl:.3f} val {vvl:.3f} sign-acc {acc:.3f} | {time.time()-t0:.0f}s", flush=True)
torch.save(net.state_dict(), a.out.replace(".onnx", ".pt")); export(net, a.out); print("exported", a.out)
