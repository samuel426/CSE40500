#!/usr/bin/env python3
# validate_onnx.py  (revised)
import os, sys, onnx
from prettytable import PrettyTable

ROOT = "./onnx_models"
SEQ_LEN, INPUT_SZ, HIDDEN = 5, 5, 32

EXPECTS = {
    "GRU": {
        "inputs": {"input": (1, SEQ_LEN, INPUT_SZ),
                   "h0"   : (1, 1, HIDDEN)},
        "need_wrb": False,
    },
    "LSTM": {
        "inputs": {"input": (1, SEQ_LEN, INPUT_SZ),
                   "h0"   : (1, 1, HIDDEN),
                   "c0"   : (1, 1, HIDDEN)},
        "need_wrb": True,
    },
    "BiLSTM": {
        "inputs": {"input": (1, SEQ_LEN, INPUT_SZ),
                   "h0"   : (2, 1, HIDDEN),
                   "c0"   : (2, 1, HIDDEN)},
        "need_wrb": True,
    },
}

def vi_shape(vi):
    return tuple(d.dim_value for d in vi.type.tensor_type.shape.dim)

def exists_in_graph(name, graph):
    return any(i.name == name for i in graph.input) or \
           any(t.name == name for t in graph.initializer)

def check_model(path, tag):
    exp = EXPECTS[tag]
    g   = onnx.load(path).graph

    # 1) 입력 이름·shape
    ok_inputs = True
    for n, shp in exp["inputs"].items():
        vi = next((i for i in g.input if i.name == n), None)
        if vi is None or vi_shape(vi) != shp:
            ok_inputs = False
            print(f"  ✗ {os.path.basename(path)} : input {n} mismatch")
    # 2) W/R/B 존재 여부
    ok_wrb = True
    if exp["need_wrb"]:
        lstm_nodes = [n for n in g.node if n.op_type == "LSTM"]
        if not lstm_nodes:
            ok_wrb = False
        else:
            # LSTM 노드의 두번째·세번째 input 이름이 W,R
            w_name, r_name = lstm_nodes[0].input[1:3]
            ok_wrb = exists_in_graph(w_name, g) and exists_in_graph(r_name, g)
        if not ok_wrb:
            print(f"  ✗ {os.path.basename(path)} : W/R not found")
    return ok_inputs and ok_wrb

def main():
    tbl = PrettyTable(["Model","File","Passed"])
    for tag in ("GRU","LSTM","BiLSTM"):
        d = os.path.join(ROOT, tag)
        if not os.path.isdir(d): continue
        for f in sorted(p for p in os.listdir(d) if p.endswith(".onnx")):
            ok = check_model(os.path.join(d,f), tag)
            tbl.add_row([tag,f,"✅" if ok else "❌"])
    print(tbl)

if __name__ == "__main__":
    if not os.path.isdir(ROOT):
        sys.exit(f"{ROOT} not found")
    main()
