# ✅ ONNX 변환 스크립트 (GRU / LSTM / BiLSTM 포함, Hailo-8 호환)
import os
import torch
import torch.nn as nn

# ---------- 공통 설정 ----------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ROOT  = "./models"
ONNX_ROOT   = "./onnx_models"
SEQ_LEN     = 10
INPUT_SIZE  = 5

TICKERS     = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]
MODEL_TYPES = ["GRU", "LSTM", "BiLSTM"]

# ---------- 모델 정의 ----------
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(INPUT_SIZE, 32, num_layers=1, batch_first=True)
        self.conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :].unsqueeze(-1).unsqueeze(-1)
        return self.conv(h)  # [B, 1, 1, 1]

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 32, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # [B, 1]

class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 16, num_layers=1, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # [B, 1]

MODEL_CLASSES = {
    "GRU": GRUModel,
    "LSTM": LSTMModel,
    "BiLSTM": BiLSTMModel
}

# ---------- 변환 함수 ----------
def convert(model_type: str, ticker: str):
    model_dir = os.path.join(MODEL_ROOT, model_type)
    onnx_dir  = os.path.join(ONNX_ROOT,  model_type)
    os.makedirs(onnx_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{ticker}.pth")
    onnx_path  = os.path.join(onnx_dir,  f"{ticker}.onnx")

    if not os.path.exists(model_path):
        print(f"❌ {model_path} not found – skip")
        return

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
        opset_version=14,
        do_constant_folding=False,
        dynamic_axes=None
    )
    print(f"✅ Converted {model_type}-{ticker} → {onnx_path}")

# ---------- 실행 ----------
def main():
    for model_type in MODEL_TYPES:
        for ticker in TICKERS:
            convert(model_type, ticker)

if __name__ == "__main__":
    main()
