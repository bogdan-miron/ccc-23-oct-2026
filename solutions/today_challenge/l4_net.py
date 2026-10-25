import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
class Block(nn.Module):
    def __init__(s, c):
        super().__init__(); s.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False); s.b1 = nn.BatchNorm2d(c)
        s.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False); s.b2 = nn.BatchNorm2d(c)
    def forward(s, x):
        y = F.relu(s.b1(s.c1(x))); y = s.b2(s.c2(y)); return F.relu(x + y)
class Net(nn.Module):
    def __init__(s, c=64, nb=5):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(16, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        s.blocks = nn.Sequential(*[Block(c) for _ in range(nb)])
        s.pol = nn.Conv2d(c, 64, 1)
        s.vc = nn.Conv2d(c, 8, 1); s.vf1 = nn.Linear(512, 64); s.vf2 = nn.Linear(64, 1)
    def forward(s, x):
        h = s.blocks(s.stem(x))
        p = s.pol(h).reshape(-1, 64, 64).transpose(1, 2).reshape(-1, 4096)   # [from, to]
        v = F.relu(s.vc(h)).flatten(1); v = torch.tanh(s.vf2(F.relu(s.vf1(v)))).squeeze(1)
        return p, v
class _Exp(nn.Module):
    def __init__(s, net): super().__init__(); s.net = net
    def forward(s, x):
        p, v = s.net(x); return p, torch.clamp(v, -1.0, 1.0)
def export(net, path):
    net.eval(); dummy = torch.zeros(2, 16, 8, 8)
    torch.onnx.export(_Exp(net), dummy, path, input_names=["observation"], output_names=["policy", "value"],
                      dynamic_axes={"observation": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
                      opset_version=17, dynamo=False, do_constant_folding=True)
    fix_shapes(path)
def d4_perms():
    """8 board symmetries: (spatial transform fn on (B,C,8,8) tensors, index map for 4096 policy)."""
    sq = np.arange(64).reshape(8, 8); out = []
    for k in range(4):
        for fl in (False, True):
            g = np.rot90(sq, k); g = g[:, ::-1] if fl else g          # g[r',c'] = old square placed at (r',c')
            m = np.empty(64, np.int64); m[g.reshape(-1)] = np.arange(64)  # old square -> new square
            inv = np.empty(4096, np.int64)                              # inv[new_idx] = old_idx
            for f in range(64):
                for t in range(64): inv[m[f] * 64 + m[t]] = f * 64 + t
            out.append((k, fl, inv))
    return out
def apply_sp(x, k, fl):
    x = torch.rot90(x, k, (2, 3)); return torch.flip(x, (3,)) if fl else x

def fix_shapes(path):
    import onnx
    m = onnx.load(path)
    for t in m.graph.output:
        if t.name == "policy":
            d = t.type.tensor_type.shape.dim[1]; d.Clear(); d.dim_value = 4096
    keep = []
    for n in m.graph.node:
        if n.op_type == "Constant":
            t = [a for a in n.attribute if a.name == "value"][0].t; t.name = n.output[0]; m.graph.initializer.append(t)
        else: keep.append(n)
    del m.graph.node[:]; m.graph.node.extend(keep)
    onnx.save(m, path)
