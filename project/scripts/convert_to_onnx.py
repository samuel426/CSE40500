import os
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ROOT = "./models"
ONNX_ROOT = "./onnx_models"

SEQ_LEN = 15
INPUT_SIZE = 5

TICKERS = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]
MODEL_TYPES = ["LSTM", "GRU", "BiLSTM"]  # ✅ BiLSTM 포함

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_SIZE, 64, num_layers=2, batch_first=True)
        self.conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :].unsqueeze(-1).unsqueeze(-1)
        return self.conv(h)

class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 16, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

MODEL_CLASSES = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "BiLSTM": BiLSTMModel,
}

def convert(model_type: str, ticker: str):
    model_dir = os.path.join(MODEL_ROOT, model_type)
    onnx_dir = os.path.join(ONNX_ROOT, model_type)
    os.makedirs(onnx_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{ticker}.pth")
    onnx_path = os.path.join(onnx_dir, f"{ticker}.onnx")

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

def main():
    for model_type in MODEL_TYPES:
        for ticker in TICKERS:
            convert(model_type, ticker)

if __name__ == "__main__":
    main()
