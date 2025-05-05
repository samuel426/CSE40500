import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common.dataset import StockDataset

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 5
SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

DATA_ROOT = "./data"
MODEL_ROOT = "./models/BiLSTM"
TICKERS = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]

# 최적화된 BiLSTM 모델
class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 16, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        forward_h = hn[0, :, :]   # forward 마지막 상태
        backward_h = hn[1, :, :]  # backward 마지막 상태
        h = torch.cat((forward_h, backward_h), dim=1)
        return self.fc(h)

def train(model, train_loader, val_loader, save_path):
    model = model.to(DEVICE)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = np.inf
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, t in train_loader:
            x, t = x.to(DEVICE), t.to(DEVICE)
            opt.zero_grad()
            out = model(x).squeeze()
            loss = crit(out, t)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, t in val_loader:
                x, t = x.to(DEVICE), t.to(DEVICE)
                out = model(x).squeeze()
                val_loss += crit(out, t).item()
        val_loss /= len(val_loader)

        print(f"[{epoch+1}/{EPOCHS}] train: {train_loss:.6f}, val: {val_loss:.6f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ saved: {save_path}")

def main():
    os.makedirs(MODEL_ROOT, exist_ok=True)

    for tk in TICKERS:
        print(f"=== {tk} ===")
        df = pd.read_csv(os.path.join(DATA_ROOT, tk, "ohlcv.csv"), index_col=0)
        df = df[pd.to_numeric(df["Open"], errors="coerce").notnull()].astype(float)

        n = len(df); s1, s2 = int(n*0.3), int(n*0.6)
        train_ds = StockDataset(df.iloc[:s1], SEQ_LEN)
        val_ds = StockDataset(df.iloc[s1:s2], SEQ_LEN)

        tr_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(val_ds, BATCH_SIZE)

        model = BiLSTMModel()
        path = os.path.join(MODEL_ROOT, f"{tk}.pth")
        train(model, tr_loader, va_loader, path)

    print("🎯 BiLSTM 학습 완료")

if __name__ == "__main__":
    main()
