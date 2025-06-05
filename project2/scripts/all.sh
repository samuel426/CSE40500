#!/bin/bash

echo "[1-1] python train_gru.py"
python train_gru.py

echo "[1-2] python train_lstm.py"
python train_lstm.py

echo "[1-3] python train_bilstm.py"
python train_bilstm.py

echo "[2] python convert_to_onnx.py"
python convert_to_onnx.py

echo "[3] ./cp_onnx.sh"
./cp_onnx.sh

echo "[4] ./export_to_har.sh"
./export_to_har.sh

echo "[5] ./optimize_har.sh"
./optimize_har.sh

echo "[6] ./compile_to_hef.sh"
./compile_to_hef.sh

echo "[7] ./parse_hef_files.sh"
./parse_hef_files.sh

echo "[8] python npu_performance_suite9.py"
python npu_performance_suite9.py

echo "[9] python onnx_model_2_performance.py"
python onnx_model_2_performance.py

echo "[10] python tensorrt_model_2_performance.py"
python tensorrt_model_2_performance.py
