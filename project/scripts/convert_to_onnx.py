# scripts/convert_to_onnx.py

import os
import torch
import torch.nn as nn



# =======================
# 1. 설정
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ROOT = "./models"
ONNX_ROOT = "./onnx_models"

SEQ_LEN = 60
INPUT_SIZE = 5

TICKERS = ["S&P500", "Apple", "NASDAQ", "Tesla", "Samsung"]
MODEL_TYPES = ["LSTM", "GRU", "BiLSTM"]


# =======================
# 2. 모델 정의들
# =======================
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(INPUT_SIZE, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class BiLSTMModel(nn.Module):
    def __init__(self):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64*2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

MODEL_CLASSES = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "BiLSTM": BiLSTMModel
}

# =======================
# 3. 변환 함수
# =======================
def convert(model_type, ticker):
    model_dir = os.path.join(MODEL_ROOT, model_type)
    onnx_dir = os.path.join(ONNX_ROOT, model_type)
    os.makedirs(onnx_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{ticker}.pth")
    onnx_path = os.path.join(onnx_dir, f"{ticker}.onnx")

    model = MODEL_CLASSES[model_type]()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dummy_input = torch.randn(1, SEQ_LEN, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13
    )
    print(f"✅ Converted {model_type}-{ticker} to ONNX: {onnx_path}")

# =======================
# 4. 메인 실행
# =======================
def main():
    for model_type in MODEL_TYPES:
        for ticker in TICKERS:
            convert(model_type, ticker)

if __name__ == "__main__":
    main()
