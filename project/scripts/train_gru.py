# scripts/train_gru.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common.dataset import StockDataset

# =======================
# 1. 설정
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 5
SEQ_LEN = 60
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

DATA_ROOT = "./data"
MODEL_ROOT = "./models/GRU"

TICKERS = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]

# =======================
# 2. GRU 모델 정의
# =======================
class GRUModel(nn.Module):
    """
    GRU → hidden → 1×1 Conv → 종가 1 값
    Hailo SDK가 Gemm을 1×1 Conv 로 인식하도록 4‑D 텐서로 변형.
    """
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(INPUT_SIZE, 64, num_layers=2, batch_first=True)
        self.conv = nn.Conv2d(64, 1, kernel_size=1)  # 1×1 conv == Linear

    def forward(self, x):
        _, h_n = self.gru(x)          # h_n shape: [num_layers, B, 64]
        h = h_n[-1]                   # [B, 64]
        h = h.unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
        y = self.conv(h)              # [B, 1, 1, 1]
        return y.squeeze()            # [B]


# =======================
# 3. 학습 함수
# =======================
def train(model, train_loader, val_loader, save_path):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = np.inf

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

        # 최적 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to {save_path}")

# =======================
# 4. 데이터 로딩 및 학습 실행
# =======================
def main():
    os.makedirs(MODEL_ROOT, exist_ok=True)

    for ticker in TICKERS:
        print(f"=== Training {ticker} ===")
        data_path = os.path.join(DATA_ROOT, ticker, "ohlcv.csv")
        df = pd.read_csv(
            data_path,
            skiprows=3,  # (3줄 무시: Ticker + Date + 공백 줄)
            header=None,  # 헤더 없다고 선언
            names=["Open", "High", "Low", "Close", "Volume"],  # 열 이름 직접 지정
            index_col=None
        )



        total_len = len(df)
        split1 = int(total_len * 0.3)
        split2 = int(total_len * 0.6)

        df_train = df.iloc[:split1]
        df_val = df.iloc[split1:split2]

        train_dataset = StockDataset(df_train)
        val_dataset = StockDataset(df_val)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = GRUModel()

        save_path = os.path.join(MODEL_ROOT, f"{ticker}.pth")
        train(model, train_loader, val_loader, save_path)

    print("🎯 모든 GRU 모델 학습 완료.")

# =======================
# 5. 시작
# =======================
if __name__ == "__main__":
    main()
