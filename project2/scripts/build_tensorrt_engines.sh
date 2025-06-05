#!/bin/bash

echo "🚀 [TensorRT 엔진 생성 및 최적화 시작]"

ONNX_MODEL_DIR="./onnx_models" # Directory containing your .onnx files
TENSORRT_ENGINE_DIR="./tensorrt_engines" # Directory to save TensorRT engines
LOG_DIR="./tensorrt_logs" # Directory to save trtexec logs

mkdir -p "$TENSORRT_ENGINE_DIR"
mkdir -p "$LOG_DIR"

# Check if ONNX model directory exists
if [ ! -d "$ONNX_MODEL_DIR" ]; then
    echo "ONNX model directory $ONNX_MODEL_DIR not found."
    exit 1
fi

# Iterate over each subdirectory in ONNX_MODEL_DIR (e.g., GRU, LSTM)
for MODEL_TYPE_DIR in "$ONNX_MODEL_DIR"/*/; do
    if [ -d "$MODEL_TYPE_DIR" ]; then
        MODEL_TYPE_NAME=$(basename "$MODEL_TYPE_DIR")
        echo "Processing model type: $MODEL_TYPE_NAME"

        # Create corresponding subdirectory in TENSORRT_ENGINE_DIR
        mkdir -p "$TENSORRT_ENGINE_DIR/$MODEL_TYPE_NAME"

        # Iterate over .onnx files in the current model type subdirectory
        for onnx_file_path in "$MODEL_TYPE_DIR"*.onnx; do
            if [ -f "$onnx_file_path" ]; then
                onnx_filename=$(basename "$onnx_file_path")
                engine_filename="${onnx_filename%.onnx}.plan" # Change extension to .plan or .engine
                engine_file_path="$TENSORRT_ENGINE_DIR/$MODEL_TYPE_NAME/$engine_filename"
                log_file_path="$LOG_DIR/${onnx_filename%.onnx}_trtexec.log"

                echo "🔄 Converting $onnx_file_path → $engine_file_path"

                # Basic trtexec command
                # Add --fp16 for FP16 precision
                # For INT8 precision, you'd need --int8 and calibration options (e.g., --calib=<calibration_file>)
                # Add --verbose for more detailed output from trtexec
                # Use --explicitBatch if your ONNX model has an explicit batch dimension
                
                trtexec --onnx="$onnx_file_path" \
                        --saveEngine="$engine_file_path" \
                        --explicitBatch \
                        # --fp16 \
                        # --int8 \
                        # --verbose \
                        > "$log_file_path" 2>&1
                
                if [ $? -eq 0 ]; then
                    echo "✅ Successfully created TensorRT engine: $engine_file_path"
                    echo "   Log file: $log_file_path"
                else
                    echo "❌ Error creating TensorRT engine for $onnx_file_path. Check log: $log_file_path"
                fi
            fi
        done
    fi
done

echo "✅ TensorRT 엔진 생성 및 최적화 완료"
