#!/bin/bash

echo "🚀 [Hailo HAR 파일 생성 시작]"

MODEL_TYPES=("GRU" "LSTM" "BiLSTM")
TICKERS=("KOSPI" "Apple" "NASDAQ" "Tesla" "Samsung")

mkdir -p ./compiled_model

for model_type in "${MODEL_TYPES[@]}"; do
    for ticker in "${TICKERS[@]}"; do
        onnx_path="./onnx_models/${model_type}/${ticker}.onnx"
        har_path="./compiled_model/${model_type}_${ticker}.har"

        if [ -f "$onnx_path" ]; then
            echo "🔄 Parsing $onnx_path → $har_path"
            hailo parser onnx "$onnx_path" \
                --net-name "${model_type}_${ticker}" \
                --hw-arch hailo8 \
                --har-path "$har_path"
        else
            echo "  ONNX 파일 없음: $onnx_path"
        fi
    done
done

echo "✅ HAR 파일 생성 완료"
