#!/usr/bin/env python3
# view_onnx_io.py
import os, onnx

MODELS = {
    "GRU"    : "onnx_models/GRU/KOSPI.onnx",
    "LSTM"   : "onnx_models/LSTM/KOSPI.onnx",
    "BiLSTM" : "onnx_models/BiLSTM/KOSPI.onnx",
}

# 가중치·바이어스 같은 initializers 이름 패턴
WEIGHT_TOKENS = ("weight", "bias", "W", "R", "B")

def is_weight(name):
    low = name.lower()
    return any(tok in low for tok in WEIGHT_TOKENS) or "onnx::" in name

def print_tensor(tensor):
    shape = [
        d.dim_value if d.HasField("dim_value") else "?"
        for d in tensor.type.tensor_type.shape.dim
    ]
    print(f"  • {tensor.name:<20}  {shape}")

for tag, path in MODELS.items():
    if not os.path.exists(path):
        print(f"❌ {tag}  : file not found — {path}")
        continue

    model = onnx.load(path)
    print(f"\n=== {tag}  ({os.path.basename(path)}) ===")

    # 1) 런타임 입력만 (가중치 제외)
    runtime_inputs = [i for i in model.graph.input if not is_weight(i.name)]
    print("Inputs:")
    for t in runtime_inputs:
        print_tensor(t)

    # 2) 출력
    print("Outputs:")
    for t in model.graph.output:
        print_tensor(t)
