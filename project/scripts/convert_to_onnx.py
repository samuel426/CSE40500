import os
import torch
from train_gru import GRUModel
from train_bilstm import BiLSTMModel
from train_lstm import LSTMModel  # 기존 LSTM 추가

DEVICE = torch.device("cpu")
SEQ_LEN = 10
INPUT_SIZE = 5
TICKERS = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]

MODEL_CLASSES = {
    "GRU": GRUModel,
    "BiLSTM": BiLSTMModel,
    "LSTM": LSTMModel,
}

def convert(model_type, ticker):
    model = MODEL_CLASSES[model_type]()
    model_path = f"./models/{model_type}/{ticker}.pth"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dummy = torch.randn(1, SEQ_LEN, INPUT_SIZE)
    onnx_dir = f"./onnx_models/{model_type}"
    os.makedirs(onnx_dir, exist_ok=True)

    onnx_path = os.path.join(onnx_dir, f"{ticker}.onnx")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        do_constant_folding=False,
        dynamic_axes=None
    )

    print(f"{model_type}-{ticker} ONNX 변환 완료 ➡️ {onnx_path}")

def main():
    for m in MODEL_CLASSES:
        for tk in TICKERS:
            convert(m, tk)

if __name__ == "__main__":
    main()
