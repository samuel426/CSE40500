import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common.dataset import StockDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 5
SEQ_LEN = 32
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

DATA_ROOT = "./data"
MODEL_ROOT = "./models/GRU"
TICKERS = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(INPUT_SIZE, 32, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(32, 1)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, h0=None):
        x = x.squeeze(1)  # (B, 1, 32, 5) → (B, 32, 5)
        out, h = self.gru(x, h0)
        out = out[:, -1, :]
        price = self.fc1(out)
        volume = self.fc2(out)
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
            h0 = torch.zeros(1, x.size(0), 32).to(DEVICE)
            optimizer.zero_grad()
            price_pred, volume_pred = model(x, h0)
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
                h0 = torch.zeros(1, x.size(0), 32).to(DEVICE)
                price_pred, volume_pred = model(x, h0)
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


def main():
    os.makedirs(MODEL_ROOT, exist_ok=True)
    for tk in TICKERS:
        csv_path = os.path.join(DATA_ROOT, tk, "ohlcv.csv")
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path, skiprows=2, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df = df.dropna()
        n = len(df)
        s1, s2 = int(n * 0.3), int(n * 0.6)
        train_ds = StockDataset(df.iloc[:s1], seq_len=SEQ_LEN)
        val_ds = StockDataset(df.iloc[s1:s2], seq_len=SEQ_LEN)
        tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        model_path = os.path.join(MODEL_ROOT, f"{tk}.pth")
        train(GRUModel(), tr_loader, va_loader, model_path)


if __name__ == "__main__":
    main()
