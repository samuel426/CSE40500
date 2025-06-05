#!/bin/bash

echo "🚀 [TensorRT 엔진 검증 및 프로파일링 시작]"

TENSORRT_ENGINE_DIR="./tensorrt_engines" # Directory where TensorRT engines are saved
LOG_DIR="./tensorrt_logs" # Directory to save verification and profiling logs
ONNX_MODEL_DIR="./onnx_models" # Directory containing original .onnx files (needed for comparison data)
# Ensure a directory for verification/profiling specific logs
VERIFY_LOG_DIR="$LOG_DIR/verification_profiling"
mkdir -p "$VERIFY_LOG_DIR"

# Check if TensorRT engine directory exists
if [ ! -d "$TENSORRT_ENGINE_DIR" ]; then
    echo "TensorRT engine directory $TENSORRT_ENGINE_DIR not found."
    echo "Please run 'build_tensorrt_engines.sh' first."
    exit 1
fi

# Iterate over each model type subdirectory in TENSORRT_ENGINE_DIR
for MODEL_TYPE_DIR in "$TENSORRT_ENGINE_DIR"/*/; do
    if [ -d "$MODEL_TYPE_DIR" ]; then
        MODEL_TYPE_NAME=$(basename "$MODEL_TYPE_DIR")
        echo "🔎 Verifying and Profiling model type: $MODEL_TYPE_NAME"

        # Iterate over .plan files (TensorRT engines)
        for engine_file_path in "$MODEL_TYPE_DIR"*.plan; do
            if [ -f "$engine_file_path" ]; then
                engine_filename=$(basename "$engine_file_path")
                log_file_path="$VERIFY_LOG_DIR/${engine_filename%.plan}_profile.log"
                verification_status_file="$VERIFY_LOG_DIR/${engine_filename%.plan}_verification_status.txt"

                echo "🔄 Profiling engine: $engine_file_path"

                # --- Profiling with trtexec ---
                # This command loads the engine and runs inference for benchmarking.
                # Adjust --iterations, --duration, --avgRuns as needed.
                # --useSpinWait can give more stable latency measurements.
                # Add --verbose for more details.
                # Input data might be required for some models, use --loadInputs
                # Example: --loadInputs=input_name:path/to/input_data.dat
                
                trtexec --loadEngine="$engine_file_path" \
                        --iterations=100 \
                        --useSpinWait \
                        --avgRuns=10 \
                        # --verbose \
                        > "$log_file_path" 2>&1
                
                if [ $? -eq 0 ]; then
                    echo "✅ Profiling complete for $engine_filename. Log: $log_file_path"
                    grep "Latency" "$log_file_path" # Display key performance metrics
                    grep "Throughput" "$log_file_path"
                else
                    echo "❌ Error during profiling for $engine_filename. Check log: $log_file_path"
                fi

                # --- Numerical Verification (Placeholder) ---
                echo "📝 Numerical verification for $engine_filename:"
                # This part typically requires a Python script.
                # The Python script would:
                # 1. Load the TensorRT engine (`.plan` file).
                # 2. Load the corresponding ONNX model (e.g., from $ONNX_MODEL_DIR/$MODEL_TYPE_NAME/${engine_filename%.plan}.onnx).
                # 3. Generate or load sample input data.
                # 4. Run inference with both the TensorRT engine and the ONNX model (using onnxruntime).
                # 5. Compare the outputs (e.g., using numpy.allclose).
                # 6. Write "SUCCESS" or "FAILURE" with details to $verification_status_file.
                
                # Example placeholder for calling a Python script:
                # python verify_engine.py --engine_path "$engine_file_path" \
                #                         --onnx_path "$ONNX_MODEL_DIR/$MODEL_TYPE_NAME/${engine_filename%.plan}.onnx" \
                #                         --output_status_file "$verification_status_file"

                echo "📝 Numerical verification for $engine_filename:"
                # Ensure the corresponding ONNX file exists
                original_onnx_file="$ONNX_MODEL_DIR/$MODEL_TYPE_NAME/${engine_filename%.plan}.onnx"
                if [ ! -f "$original_onnx_file" ]; then
                    echo "   Corresponding ONNX file not found: $original_onnx_file. Skipping verification."
                    echo "SKIPPED: ONNX file missing" > "$verification_status_file"
                else
                    python3 ./verify_engine.py --engine_path "$engine_file_path" \
                                            --onnx_path "$original_onnx_file" \
                                            --output_status_file "$verification_status_file" \
                                            --rtol 1e-3 --atol 1e-5 # Adjust tolerances as needed
                                            # Add --onnx_gpu if you want to try ONNX on GPU
                fi

                if [ -f "$verification_status_file" ] && grep -q "SUCCESS" "$verification_status_file"; then
                    echo "✅ Numerical verification successful (based on $verification_status_file)."
                elif [ -f "$verification_status_file" ];
                    echo "❌ Numerical verification FAILED or status unclear. Check $verification_status_file."
                else
                    echo "⚠️ Python verification script not run or status file not found for $engine_filename. Manual check needed."
                    echo "   Expected status file: $verification_status_file"
                fi
                echo "-----------------------------------"
            fi
        done
    fi
done

echo "✅ TensorRT 엔진 검증 및 프로파일링 단계 완료 (부분적 자동화)"
echo "   (수치적 정확성 검증은 별도의 Python 스크립트 구현 및 연동 필요)"
