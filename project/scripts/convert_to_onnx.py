# scripts/convert_to_onnx.py
"""
ONNX 변환 스크립트 (Hailo‑8 호환)
--------------------------------------------------
* **Bi‑LSTM 제외** – SDK 3.31 기준 미지원
* **LSTM SEQ_LEN 15** – unroll 레이어 < 300 개로 제한
"""
import os
import torch
import torch.nn as nn

# =======================
# 1. 공통 설정
# =======================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ROOT  = "./models"      # .pth 저장 폴더
ONNX_ROOT   = "./onnx_models" # .onnx 출력 폴더

SEQ_LEN     = 15  # ⚠ LSTM unroll 제한에 맞춰 15 step
INPUT_SIZE  = 5   # OHLCV

TICKERS     = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]
MODEL_TYPES = ["LSTM", "GRU"]  # ✅ BiLSTM 제거

# =======================
# 2. 모델 정의
# =======================
class LSTMModel(nn.Module):
    """ 2‑layer LSTM → 마지막 hidden → Linear(64→1) """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 64, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)     # out: [B, T, 64]
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    """ 2‑layer GRU + 1×1 Conv to satisfy Hailo parser """
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(INPUT_SIZE, 64, num_layers=2, batch_first=True)
        self.conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        _, h_n = self.gru(x)      # h_n: [layers, B, 64]
        h = h_n[-1].unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
        return self.conv(h).squeeze()            # [B]

MODEL_CLASSES = {
    "LSTM": LSTMModel,
    "GRU":  GRUModel,
}

# =======================
# 3. 변환 함수
# =======================

def convert(model_type: str, ticker: str):
    model_dir = os.path.join(MODEL_ROOT, model_type)
    onnx_dir  = os.path.join(ONNX_ROOT,  model_type)
    os.makedirs(onnx_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{ticker}.pth")
    onnx_path  = os.path.join(onnx_dir,  f"{ticker}.onnx")

    if not os.path.exists(model_path):
        print(f"❌ {model_path} not found – skip")
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
        opset_version=14,          # Hailo 3.31 호환
        do_constant_folding=False, # LSTM 파라미터 보존
        dynamic_axes=None
    )
    print(f"✅ Converted {model_type}‑{ticker} → {onnx_path}")

# =======================
# 4. 메인
# =======================

def main():
    for model_type in MODEL_TYPES:
        for ticker in TICKERS:
            convert(model_type, ticker)

if __name__ == "__main__":
    main()
