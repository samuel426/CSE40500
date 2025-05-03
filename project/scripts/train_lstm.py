# scripts/train_lstm.py

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
SEQ_LEN = 15        # 두 파일 모두
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

DATA_ROOT = "./data"
MODEL_ROOT = "./models/LSTM"
TICKERS = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 학습 함수
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to {save_path}")

# 데이터 로딩 및 학습 실행
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

        model = LSTMModel()

        save_path = os.path.join(MODEL_ROOT, f"{ticker}.pth")
        train(model, train_loader, val_loader, save_path)

    print("🎯 모든 LSTM 모델 학습 완료.")

if __name__ == "__main__":
    main()
