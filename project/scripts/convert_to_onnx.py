import os
import torch
import numpy as np
import onnx
from onnxsim import simplify
from train_gru import GRUModel
from train_lstm import LSTMModel
from train_bilstm import BiLSTMModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ROOT = "./models"
ONNX_ROOT = "./onnx_models"

SEQ_LEN_GRU = 32
SEQ_LEN_LSTM = 32
SEQ_LEN_BiLSTM = 32

INPUT_SIZE_GRU = 5
INPUT_SIZE_LSTM = 5
INPUT_SIZE_BiLSTM = 5

def export_onnx(model, model_name, output_dir, input_tensor):
    """모델을 ONNX 형식으로 변환하고 Simplifier 적용"""
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    try:
        # ONNX Export
        torch.onnx.export(
            model,
            input_tensor,
            onnx_path,
            input_names=["input"],
            output_names=["output1", "output2"],
            opset_version=14,
            export_params=True
        )
        print(f"✅ ONNX model exported: {onnx_path}")

        # ONNX Simplifier 적용
        try:
            model_onnx = onnx.load(onnx_path)
            model_simp, check = simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simp, onnx_path)
            print(f"✅ ONNX model simplified: {onnx_path}")
        except Exception as e:
            print(f"❌ Simplification failed: {e}")

    except Exception as e:
        print(f"❌ ONNX export failed: {e}")


def process_model(model_class, model_dir, output_dir, input_tensor):
    """디렉터리 내의 모든 모델(.pth)을 ONNX로 변환"""
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model_name = os.path.splitext(model_file)[0]

        # 모델 초기화 및 가중치 로드
        model = model_class().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        print(f"=== Exporting {model_name} ===")
        export_onnx(model, model_name, output_dir, input_tensor)


def main():
    os.makedirs(ONNX_ROOT, exist_ok=True)

    # GRU 모델들 변환
    gru_dir = os.path.join(MODEL_ROOT, "GRU")
    gru_output_dir = os.path.join(ONNX_ROOT, "GRU")
    if os.path.exists(gru_dir):
        # GRU의 입력 텐서 (1, 1, 32, 5)
        input_tensor_gru = torch.randn(1, 1, SEQ_LEN_GRU, INPUT_SIZE_GRU).to(DEVICE)
        process_model(GRUModel, gru_dir, gru_output_dir, input_tensor_gru)

    # LSTM 모델들 변환
    lstm_dir = os.path.join(MODEL_ROOT, "LSTM")
    lstm_output_dir = os.path.join(ONNX_ROOT, "LSTM")
    if os.path.exists(lstm_dir):
        # LSTM의 입력 텐서 (1, 1, 32, 5)
        input_tensor_lstm = torch.randn(1, 1, SEQ_LEN_LSTM, INPUT_SIZE_LSTM).to(DEVICE)
        process_model(LSTMModel, lstm_dir, lstm_output_dir, input_tensor_lstm)

    # BiLSTM 모델들 변환
    bilstm_dir = os.path.join(MODEL_ROOT, "BiLSTM")
    bilstm_output_dir = os.path.join(ONNX_ROOT, "BiLSTM")
    if os.path.exists(bilstm_dir):
        # BiLSTM의 입력 텐서 (1, 1, 32, 5)
        input_tensor_bilstm = torch.randn(1, 1, SEQ_LEN_BiLSTM, INPUT_SIZE_BiLSTM).to(DEVICE)
        process_model(BiLSTMModel, bilstm_dir, bilstm_output_dir, input_tensor_bilstm)


if __name__ == "__main__":
    main()
