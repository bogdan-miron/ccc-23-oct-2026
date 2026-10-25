"""Tiny hand-built policy/value ONNX (~40 KB). Usage: python build_small.py out.onnx [valscale] [capscale]
value = tanh(linear(material diff + kernel centrality)); policy = capture prior by victim type, gathered to from*64+to."""
import sys, numpy as np, onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
out = sys.argv[1]; vs = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35; cs = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
MAT = {0: 1.0, 1: 2.5, 2: 1.7, 3: 1.6, 4: 2.8}; CAP = [2.0, 3.0, 3.0, 2.5, 3.5, 6.0]; RING_K = 0.12
def ring(r, c): return min(r, c, 7 - r, 7 - c)
Wv = np.zeros((1024, 1), np.float32)
for r in range(8):
    for c in range(8):
        s = r * 8 + c
        for k, w in MAT.items(): Wv[k * 64 + s, 0] += w * vs; Wv[(6 + k) * 64 + s, 0] -= w * vs
        Wv[5 * 64 + s, 0] += RING_K * ring(r, c) * vs / 0.35; Wv[11 * 64 + s, 0] -= RING_K * ring(r, c) * vs / 0.35
Wc = (np.array(CAP, np.float32) * cs).reshape(1, 6, 1, 1)
I = lambda a, n: nh.from_array(np.asarray(a), n)
inits = [I(Wv, "Wv"), I(np.zeros(1, np.float32), "bv"), I(Wc, "Wc"), I(np.array([-1], np.int64), "shp1"),
         I(np.array([0, 64], np.int64), "shp64"), I(np.array([6], np.int64), "st"),
         I(np.array([12], np.int64), "en"), I(np.array([1], np.int64), "ax")]
nodes = [h.make_node("Flatten", ["observation"], ["flat"], axis=1),
         h.make_node("Gemm", ["flat", "Wv", "bv"], ["v1"]), h.make_node("Tanh", ["v1"], ["v2"]),
         h.make_node("Reshape", ["v2", "shp1"], ["value"]),
         h.make_node("Slice", ["observation", "st", "en", "ax"], ["enemy"]),
         h.make_node("Conv", ["enemy", "Wc"], ["cap"], kernel_shape=[1, 1]),
         h.make_node("Reshape", ["cap", "shp64"], ["g"]),
         h.make_node("Concat", ["g"] * 64, ["policy"], axis=1)]
g = h.make_graph(nodes, "hand_small",
    [h.make_tensor_value_info("observation", T.FLOAT, ["batch", 16, 8, 8])],
    [h.make_tensor_value_info("policy", T.FLOAT, ["batch", 4096]), h.make_tensor_value_info("value", T.FLOAT, ["batch"])], inits)
m = h.make_model(g, opset_imports=[h.make_opsetid("", 13)]); m.ir_version = 8
onnx.checker.check_model(m); onnx.save(m, out); print("saved", out)
