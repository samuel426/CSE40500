import os
import torch
from train_gru import GRUModel
from train_bilstm import BiLSTMModel
from train_lstm import LSTMModel

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
    path = f"./models/{model_type}/{ticker}.pth"
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()

    dummy_input = torch.randn(1, SEQ_LEN, INPUT_SIZE)
    onnx_path = f"./onnx_models/{model_type}/{ticker}.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=14,
        export_params=True
    )

    print(f"✅ ONNX export 완료: {onnx_path}")

def main():
    for m in MODEL_CLASSES:
        for tk in TICKERS:
            convert(m, tk)

if __name__ == "__main__":
    main()
