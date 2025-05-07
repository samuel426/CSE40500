# train_lstm.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common.dataset import StockDataset

# ---------- 설정 ----------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE  = 5
SEQ_LEN     = 10
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 0.001

DATA_ROOT   = "./data"
MODEL_ROOT  = "./models/LSTM"
TICKERS     = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]

# ---------- LSTM 모델 (Hailo 호환) ----------
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 64, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]            # 마지막 타임스텝의 출력
        return self.fc(out)

# ---------- 학습 루프 ----------
def train(model, train_loader, val_loader, save_path):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')
    for epoch in range(1, EPOCHS+1):
        # -- train --
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # -- val --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).squeeze()
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        print(f"[{epoch:02}/{EPOCHS}] train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ saved {save_path}")

# ---------- 실행 ----------
def main():
    os.makedirs(MODEL_ROOT, exist_ok=True)

    for tk in TICKERS:
        print(f"=== {tk} ===")
        # CSV 로드: 첫 두 줄 스킵, 세 번째 줄부터 Date 포함
        df = pd.read_csv(
            os.path.join(DATA_ROOT, tk, "ohlcv.csv"),
            skiprows=2,
            names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        df.set_index('Date', inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # 30%-30%-40% 분할
        n = len(df)
        s1, s2 = int(n*0.3), int(n*0.6)

        train_ds = StockDataset(df.iloc[:s1], seq_len=SEQ_LEN)
        val_ds   = StockDataset(df.iloc[s1:s2], seq_len=SEQ_LEN)
        test_ds  = StockDataset(df.iloc[s2:], seq_len=SEQ_LEN)

        tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE)
        te_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE)

        # 모델 학습
        model_path = os.path.join(MODEL_ROOT, f"{tk}.pth")
        train(LSTMModel(), tr_loader, va_loader, model_path)

        # 테스트셋 평가
        model = LSTMModel().to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in te_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).squeeze()
                test_loss += nn.MSELoss()(pred, y).item()
        test_loss /= len(te_loader)
        print(f"--- Test Loss for {tk}: {test_loss:.6f} ---\n")

    print("🎯 LSTM Training & Testing Complete")

if __name__ == "__main__":
    main()
