import os
import torch
from train_gru import GRUModel
from train_bilstm import BiLSTMModel
from train_lstm import LSTMModel

DEVICE = torch.device("cpu")
SEQ_LEN = 10
INPUT_SIZE = 5
MODEL_ROOT = "./models"
ONNX_ROOT = "./onnx_models"
TICKERS = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]

# 모델 클래스와 경로 매핑
MODEL_CLASSES = {
    "GRU": GRUModel,
    "BiLSTM": BiLSTMModel,
    "LSTM": LSTMModel
}

def load_model(model_class, model_path):
    """ 모델 클래스와 경로를 받아 모델 인스턴스 생성 및 파라미터 로드 """
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def convert(model, model_name, ticker):
    """ ONNX 변환 함수 """
    dummy_input = torch.randn(1, SEQ_LEN, INPUT_SIZE).to(DEVICE)
    save_dir = os.path.join(ONNX_ROOT, model_name)
    os.makedirs(save_dir, exist_ok=True)
    onnx_path = os.path.join(save_dir, f"{ticker}.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        export_params=True
    )
    print(f"✅ ONNX export 완료: {onnx_path}")

def main():
    os.makedirs(ONNX_ROOT, exist_ok=True)

    for model_name, model_class in MODEL_CLASSES.items():
        print(f"=== {model_name} ===")
        for ticker in TICKERS:
            model_path = os.path.join(MODEL_ROOT, model_name, f"{ticker}.pth")
            if os.path.exists(model_path):
                print(f"🔄 {ticker} 모델 로드 중: {model_path}")
                model = load_model(model_class, model_path)
                convert(model, model_name, ticker)
            else:
                print(f"❌ 모델 파일이 없습니다: {model_path}")

if __name__ == "__main__":
    main()
