import os, torch, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from common.dataset import StockDataset
from hyp import *



MODEL_ROOT = "./models/LSTM"
os.makedirs(MODEL_ROOT, exist_ok=True)


class PriceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(HIDDEN_SIZE, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):                       # [N,1,seq,feat]
        x = x.squeeze(1)                        # [N,seq,feat]
        out, _ = self.lstm(x)
        out = out[:, -1, :].unsqueeze(-1)       # [N, hidden, 1]
        return self.head(out).squeeze(-1)       # [N,1]


def train(model, tr_loader, va_loader, save_file):
    model.to(DEVICE)
    loss_fn, opt, best = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=LR), 1e9
    for ep in range(1, EPOCHS + 1):
        model.train(); tr_loss = 0.0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = loss_fn(model(x).squeeze(-1), y)
            opt.zero_grad(); loss.backward(); opt.step(); tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                va_loss += loss_fn(model(x).squeeze(-1), y)
        va_loss /= len(va_loader)
        print(f"[{ep:02}/{EPOCHS}] train={tr_loss:.4f}  val={va_loss:.4f}")
        if va_loss < best:
            best = va_loss; torch.save(model.state_dict(), save_file)


def main():
    for tk in TICKERS:
        csv = os.path.join(DATA_ROOT, tk, "ohlcv.csv")
        if not os.path.exists(csv):
            print("❌", csv); continue
        df = pd.read_csv(csv, skiprows=3, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']).dropna()
        n = len(df); s1, s2 = int(n * .3), int(n * .6)
        tr_loader = DataLoader(StockDataset(df.iloc[:s1], SEQ_LEN), BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(StockDataset(df.iloc[s1:s2], SEQ_LEN), BATCH_SIZE)
        train(PriceLSTM(), tr_loader, va_loader, os.path.join(MODEL_ROOT, f"{tk}.pth"))


if __name__ == "__main__":
    main()
