import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common.dataset import StockDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 5
SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

DATA_ROOT = "./data"
MODEL_ROOT = "./models/GRU"
TICKERS = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(SEQ_LEN * INPUT_SIZE, 64)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(64, 32, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        B, T, C = x.shape
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = x.view(B, 1, -1)
        out, _ = self.gru(x)
        return self.fc2(out[:, -1, :])

# 학습 함수는 기존과 동일
# main 함수도 동일하게 ticker별 학습 및 저장


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

        model = GRUModel()
        path = os.path.join(MODEL_ROOT, f"{tk}.pth")
        train(model, tr_loader, va_loader, path)

    print("🎯 GRU 학습 완료")

if __name__ == "__main__":
    main()
