# scripts/train_gru.py
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
SEQ_LEN     = 15          # ✔ 15‑step
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 0.001

DATA_ROOT   = "./data"
MODEL_ROOT  = "./models/GRU"
TICKERS     = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]

# ---------- GRU 모델 ----------
class GRUModel(nn.Module):
    """
    GRU → 1×1 Conv (Linear) → [B,1,1,1]  **squeeze 안함** (N,C,H,W 형태 유지)
    """
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(INPUT_SIZE, 64, num_layers=2, batch_first=True)
        self.conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):                       # x : [B,15,5]
        _, h_n = self.gru(x)                    # h_n: [2,B,64]
        h = h_n[-1].unsqueeze(-1).unsqueeze(-1) # [B,64,1,1]
        y = self.conv(h)                        # [B,1,1,1]
        return y                                # 그대로 반환

# ---------- 학습 루프 ----------
def train(model, train_loader, val_loader, save_path):
    model = model.to(DEVICE)
    crit  = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    best = np.inf
    for epoch in range(EPOCHS):
        # --- train ---
        model.train(); train_loss = 0
        for x, t in train_loader:
            x, t = x.to(DEVICE), t.to(DEVICE)
            opt.zero_grad()
            out = model(x).squeeze()
            loss = crit(out, t)
            loss.backward(); opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- val ---
        model.eval(); val_loss = 0
        with torch.no_grad():
            for x, t in val_loader:
                x, t = x.to(DEVICE), t.to(DEVICE)
                out = model(x).squeeze()
                val_loss += crit(out, t).item()
        val_loss /= len(val_loader)

        print(f"[{epoch+1:02}/{EPOCHS}] train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ saved: {save_path}")

# ---------- 실행 ----------
def main():
    os.makedirs(MODEL_ROOT, exist_ok=True)

    for tk in TICKERS:
        print(f"=== {tk} ===")
        df = pd.read_csv(os.path.join(DATA_ROOT, tk, "ohlcv.csv"), index_col=0)  # ✔ skiprows 제거

        # ➡ 숫자 아닌 행 삭제 (Ticker 문자열 등)
        df = df[pd.to_numeric(df["Open"], errors="coerce").notnull()].astype(float)
        
        # 30‑30‑40 split
        n = len(df); s1, s2 = int(n*0.3), int(n*0.6)
        train_ds = StockDataset(df.iloc[:s1], seq_len=SEQ_LEN)
        val_ds   = StockDataset(df.iloc[s1:s2], seq_len=SEQ_LEN)

        tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = GRUModel()
        path  = os.path.join(MODEL_ROOT, f"{tk}.pth")
        train(model, tr_loader, va_loader, path)

    print("🎯 GRU 학습 완료")

if __name__ == "__main__":
    main()
