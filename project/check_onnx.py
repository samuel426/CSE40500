import os
import onnx
from onnx import checker, shape_inference

def check_onnx_model(onnx_path):
    try:
        print(f"🔍 Checking ONNX model: {onnx_path}")
        # 모델 로드
        model = onnx.load(onnx_path)

        # Shape inference (입출력 텐서의 형상을 추론하여 추가)
        model = shape_inference.infer_shapes(model)

        # 모델 유효성 검사
        checker.check_model(model)
        print("✅ Model is valid.")

    except onnx.checker.ValidationError as e:
        print(f"❌ Model validation failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error during model checking: {e}")

def main():
    ONNX_ROOT = "./onnx_models"
    
    # ONNX 폴더 내의 모든 .onnx 파일 확인
    for root, _, files in os.walk(ONNX_ROOT):
        for file in files:
            if file.endswith(".onnx"):
                onnx_path = os.path.join(root, file)
                check_onnx_model(onnx_path)

if __name__ == "__main__":
    main()

