#!/bin/bash

echo "🚀 [Hailo HAR 파일 생성 시작]"

MODEL_TYPES=("GRU")
MODEL_TYPES=("GRU" "LSTM" "BiLSTM")
TICKERS=("KOSPI" "Apple" "NASDAQ" "Tesla" "Samsung")
TICKERS=("Apple")

# 각 모델 타입별 end node names를 설정
get_end_node_names_param() {
    case "$1" in
        "GRU")
            echo "--end-node-names /volume_conv_out /price_conv_out"
            ;;
        "LSTM")
            echo "--end-node-names /fc_volume/Gemm /fc_price/Gemm"
            ;;
        "BiLSTM")
            echo "--end-node-names /fc_volume/Gemm /fc_price/Gemm"
            ;;
        *)
            echo ""
            ;;
    esac
}

mkdir -p ./compiled_model

for model_type in "${MODEL_TYPES[@]}"; do
    END_NODE_NAMES_PARAM=$(get_end_node_names_param "$model_type")
    for ticker in "${TICKERS[@]}"; do
        onnx_path="./onnx_models/${model_type}/${ticker}.onnx"
        har_path="./compiled_model/${model_type}_${ticker}.har"

        if [ -f "$onnx_path" ]; then
            echo "🔄 Parsing $onnx_path → $har_path"
            hailo parser onnx "$onnx_path" \
                --net-name "${model_type}_${ticker}" \
                --hw-arch hailo8l \
                --har-path "$har_path" \
                $END_NODE_NAMES_PARAM
        else
            echo "  ONNX 파일 없음: $onnx_path"
        fi
    done
done

echo "✅ HAR 파일 생성 완료"
