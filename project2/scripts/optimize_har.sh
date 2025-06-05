#!/bin/bash

echo "🚀 [Hailo HAR 파일 최적화 시작]"

MODEL_TYPES=("GRU" "LSTM" "BiLSTM")
MODEL_TYPES=("GRU" "LSTM")
TICKERS=("KOSPI" "Apple" "NASDAQ" "Tesla" "Samsung" "SnP500")
TICKERS=("SnP500")

mkdir -p ./compiled_model/optimized

for model_type in "${MODEL_TYPES[@]}"; do
    for ticker in "${TICKERS[@]}"; do
        har_path="./compiled_model/${model_type}_${ticker}.har"
        optimized_har_path="./compiled_model/optimized/${model_type}_${ticker}_optimized.har"

        if [ -f "$har_path" ]; then
            echo "🔄 Optimizing $har_path → $optimized_har_path"
            hailo optimize "$har_path" \
                --output "$optimized_har_path" \
                --use-random-calib-set \
                --hw-arch hailo8l
        else
            echo "  HAR 파일 없음: $har_path"
        fi
    done
done

echo "✅ HAR 파일 최적화 완료"