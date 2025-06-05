#!/bin/bash

echo "🚀 [Hailo HEF 파일 컴파일 시작]"

MODEL_TYPES=("GRU" "LSTM" "BiLSTM")
MODEL_TYPES=("GRU" "LSTM")
TICKERS=("KOSPI" "Apple" "NASDAQ" "Tesla" "Samsung")
TICKERS=("Apple")

mkdir -p ./compiled_model/hef

for model_type in "${MODEL_TYPES[@]}"; do
    for ticker in "${TICKERS[@]}"; do
        
        # Determine the correct HAR path and optimization level for the current model type
        # For HEF compilation targeting hardware, we should always use the *optimized* (quantized) HAR.
        har_path_to_compile="./compiled_model/optimized/${model_type}_${ticker}_optimized.har"
        current_compiler_optimization_level=1 # Default compiler optimization level

        if [ "$model_type" = "GRU" ] || [ "$model_type" = "LSTM" ]; then
            current_compiler_optimization_level=2 #level 2 is max, 1 is medium, 0 is low
            # Ensure we are using the optimized HAR (already set in har_path_to_compile)
        elif [ "$model_type" = "BiLSTM" ]; then
            # Use compiler optimization level 0 as intended
            current_compiler_optimization_level=0
            # CRITICAL: Still must use the *optimized* (quantized) HAR.
            # The har_path_to_compile variable already points to the optimized HAR.
        fi

        if [ -f "$har_path_to_compile" ]; then
            echo "🔄 Compiling $har_path_to_compile → ./compiled_model/hef/"
            
            hailo compiler "$har_path_to_compile" \
                            --hw-arch hailo8l \
                            --output-dir ./compiled_model/hef \
                            --model-script "performance_param(compiler_optimization_level=${current_compiler_optimization_level})"
        else
            echo "  HAR 파일 없음: $har_path_to_compile. Make sure 'optimize_har.sh' was run."
        fi
    done
done

echo "✅ HEF 파일 컴파일 완료"