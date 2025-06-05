import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common.dataset import StockDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 5
SEQ_LEN = 6
HIDDEN_SIZE = 5   # Hailo에서 in_channels=out_channels, 5 이어야 함
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

DATA_ROOT = "./data"
MODEL_ROOT = "./models/BiLSTM"
TICKERS = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]
TICKERS = ["Apple"]

# 1. BiLSTM을 직접 Unroll 방식(정방향/역방향 LSTM 두 개)으로 구현할 경우:
class BiLSTMModel(nn.Module):
    def __init__(self):
        super(BiLSTMModel, self).__init__()
        self.lstm_f = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.lstm_b = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(10, 1)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x, h0_f, c0_f, h0_b, c0_b):
        x = x.squeeze(1)
        # Forward
        _, (h_f, _) = self.lstm_f(x, (h0_f, c0_f))
        # Backward
        x_rev = torch.flip(x, [1])
        _, (h_b, _) = self.lstm_b(x_rev, (h0_b, c0_b))
        h = torch.cat([h_f[-1], h_b[-1]], dim=1)
        price = self.fc1(h)
        volume = self.fc2(h)
        return price, volume

      
def train(model, train_loader, val_loader, save_path):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # 각각 (1, batch, 5)로 초기화
            h0_f = torch.zeros(1, x.size(0), 5).to(DEVICE)
            c0_f = torch.zeros(1, x.size(0), 5).to(DEVICE)
            h0_b = torch.zeros(1, x.size(0), 5).to(DEVICE)
            c0_b = torch.zeros(1, x.size(0), 5).to(DEVICE)

            optimizer.zero_grad()
            price_pred, volume_pred = model(x, h0_f, c0_f, h0_b, c0_b)
            loss1 = criterion(price_pred.squeeze(), y[:, 0])
            loss2 = criterion(volume_pred.squeeze(), y[:, 1])
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                h0_f = torch.zeros(1, x.size(0), 5).to(DEVICE)
                c0_f = torch.zeros(1, x.size(0), 5).to(DEVICE)
                h0_b = torch.zeros(1, x.size(0), 5).to(DEVICE)
                c0_b = torch.zeros(1, x.size(0), 5).to(DEVICE)
                price_pred, volume_pred = model(x, h0_f, c0_f, h0_b, c0_b)
                loss1 = criterion(price_pred.squeeze(), y[:, 0])
                loss2 = criterion(volume_pred.squeeze(), y[:, 1])
                loss = loss1 + loss2
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"[{epoch:02}/{EPOCHS}] train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved: {save_path}")

def minmax_normalize(df, columns):
    """Min-Max 정규화"""
    return (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min() + 1e-8)

def main():
    os.makedirs(MODEL_ROOT, exist_ok=True)
    for tk in TICKERS:
        csv_path = os.path.join(DATA_ROOT, tk, "ohlcv.csv")
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path, skiprows=2, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df = df.dropna()

        # === 정규화 추가 ===
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[feature_cols] = minmax_normalize(df, feature_cols)

        n = len(df)
        s1, s2 = int(n * 0.3), int(n * 0.6)
        train_ds = StockDataset(df.iloc[:s1], seq_len=SEQ_LEN)
        val_ds = StockDataset(df.iloc[s1:s2], seq_len=SEQ_LEN)
        tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        model_path = os.path.join(MODEL_ROOT, f"{tk}.pth")
        train(BiLSTMModel(), tr_loader, va_loader, model_path)


if __name__ == "__main__":
    main()
