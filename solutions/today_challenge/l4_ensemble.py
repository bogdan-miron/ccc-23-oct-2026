"""python ensemble.py out.onnx a.onnx b.onnx [wa] -> one ONNX averaging policy logits and values of two models"""
import sys, onnx, numpy as np
from onnx import helper as h, TensorProto as T, numpy_helper as nh
out, pa, pb = sys.argv[1:4]; wa = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
def prefixed(path, pre):
    m = onnx.load(path); g = m.graph; io = {i.name for i in g.input} | {o.name for o in g.output}
    ren = lambda n: n if n == "" else ("observation" if n == g.input[0].name else pre + n)
    nodes = []
    for k, n in enumerate(g.node):
        n2 = onnx.NodeProto(); n2.CopyFrom(n); n2.name = pre + (n.name or "n") + "_" + str(k)
        del n2.input[:]; n2.input.extend([ren(x) for x in n.input]); del n2.output[:]; n2.output.extend([pre + x for x in n.output]); nodes.append(n2)
    inits = []
    for i in g.initializer:
        t = onnx.TensorProto(); t.CopyFrom(i); t.name = pre + i.name; inits.append(t)
    return m, nodes, inits, pre + g.output[0].name, pre + g.output[1].name
ma, na, ia, pola, vala = prefixed(pa, "a_"); mb, nb, ib, polb, valb = prefixed(pb, "b_")
op = max(next(o.version for o in ma.opset_import if o.domain == ""), next(o.version for o in mb.opset_import if o.domain == ""))
c = lambda name, v: nh.from_array(np.array(v, np.float32), name)
extra = [c("wa", wa), c("wb", 1 - wa), c("lo", -1.0), c("hi", 1.0)]
nodes = na + nb + [
    h.make_node("Mul", [pola, "wa"], ["pa_w"]), h.make_node("Mul", [polb, "wb"], ["pb_w"]), h.make_node("Add", ["pa_w", "pb_w"], ["policy"]),
    h.make_node("Mul", [vala, "wa"], ["va_w"]), h.make_node("Mul", [valb, "wb"], ["vb_w"]), h.make_node("Add", ["va_w", "vb_w"], ["v_sum"]),
    h.make_node("Clip", ["v_sum", "lo", "hi"], ["value"])]
g = h.make_graph(nodes, "ensemble", [h.make_tensor_value_info("observation", T.FLOAT, ["batch", 16, 8, 8])],
                 [h.make_tensor_value_info("policy", T.FLOAT, ["batch", 4096]), h.make_tensor_value_info("value", T.FLOAT, ["batch"])], ia + ib + extra)
m = h.make_model(g, opset_imports=[h.make_opsetid("", op)]); m.ir_version = max(ma.ir_version, mb.ir_version)
onnx.checker.check_model(m); onnx.save(m, out); print("saved", out, "opset", op, "ir", m.ir_version)
