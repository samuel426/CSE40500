#!/bin/bash

# Hailo parser 자동 실행 스크립트
# 변환 대상: GRU, LSTM, BiLSTM 각각의 onnx 모델

MODELS=("GRU" "LSTM" "BiLSTM")
TICKERS=("KOSPI" "Apple" "NASDAQ" "Tesla" "Samsung")
ARCH="hailo8"

for model in "${MODELS[@]}"; do
  for ticker in "${TICKERS[@]}"; do
    ONNX_PATH="./onnx_models/${model}/${ticker}.onnx"
    HAR_PATH="./compiled_model/${model,,}_${ticker,,}.har"  # 소문자

    if [ -f "$ONNX_PATH" ]; then
      echo "🔄 Parsing $ONNX_PATH → $HAR_PATH"
      hailo parser onnx "$ONNX_PATH" \
        --net-name "${model,,}_${ticker,,}" \
        --hw-arch $ARCH \
        --har-path "$HAR_PATH"
    else
      echo "❌ File not found: $ONNX_PATH"
    fi
  done
done
