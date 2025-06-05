import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common.dataset import StockDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 5
SEQ_LEN = 8
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

DATA_ROOT = "./data"
MODEL_ROOT = "./models/LSTM"
TICKERS = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]
TICKERS = ["Apple"]

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h, c):
        gates = self.W_ih(x) + self.W_hh(h)
        i, f, g, o = gates.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.seq_len = seq_len
        self.fc1 = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None, c0=None):
        batch = x.size(0)
        h = h0 if h0 is not None else torch.zeros(batch, 5).to(x.device)
        c = c0 if c0 is not None else torch.zeros(batch, 5).to(x.device)
        for t in range(self.seq_len):
            h, c = self.cell(x[:, t, :], (h, c))
        price = self.fc1(h)
        volume = self.fc2(h)
        return price, volume

class LSTMModel_org(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, 5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(5, 1)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = x.squeeze(1)
        # Remove explicit h0, c0; let LSTM use default zero states
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        price = self.fc1(out)
        volume = self.fc2(out)
        return price, volume

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        x = x.squeeze(1)
        if h0 is None or c0 is None:
            # If not provided, initialize to zeros
            batch_size = x.size(0)
            num_layers = self.lstm.num_layers
            hidden_size = self.lstm.hidden_size
            h0 = torch.zeros(num_layers, batch_size, hidden_size, device=x.device)
            c0 = torch.zeros(num_layers, batch_size, hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        price = self.fc1(out)
        volume = self.fc2(out)
        return price, volume


def train_org(model, train_loader, val_loader, save_path):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Remove h0, c0
            optimizer.zero_grad()
            price_pred, volume_pred = model(x)
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
                price_pred, volume_pred = model(x)
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

def train(model, train_loader, val_loader, save_path):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            price_pred, volume_pred = model(x)
            loss1 = criterion(price_pred.squeeze(), y[:, 0])
            loss2 = criterion(volume_pred.squeeze(), y[:, 1])
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                price_pred, volume_pred = model(x)
                loss1 = criterion(price_pred.squeeze(), y[:, 0])
                loss2 = criterion(volume_pred.squeeze(), y[:, 1])
                loss = loss1 + loss2
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"[{epoch:02}/{EPOCHS}] train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved: {save_path}")

    # 에폭별 손실 기록 저장 (CSV 파일로 저장 예시)
    import pandas as pd
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(
        save_path.replace('.pth', '_loss_log.csv'), index_label='epoch'
    )

    # (선택) 추론 속도 측정 예시
    import time
    model.eval()
    with torch.no_grad():
        x, y = next(iter(val_loader))
        x = x.to(DEVICE)
        start = time.time()
        for _ in range(100):
            _ = model(x)
        elapsed = (time.time() - start) / 100
        print(f"평균 추론 시간(1배치): {elapsed*1000:.2f} ms")


def minmax_normalize(df, columns):
    """Min-Max 정규화"""
    return (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min() + 1e-8)

def evaluate(model, test_loader, save_path):
    """저장된 모델로 테스트셋 평가 및 예측/실제값 비교, 리소스 모니터링"""
    import torch
    import pandas as pd
    import time
    import psutil
    import GPUtil

    model = model.to(DEVICE)
    model.eval()
    preds_price = []
    preds_volume = []
    trues_price = []
    trues_volume = []

    # 모델 파라미터 로드
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            price_pred, volume_pred = model(x)
            preds_price.extend(price_pred.squeeze().cpu().numpy())
            preds_volume.extend(volume_pred.squeeze().cpu().numpy())
            trues_price.extend(y[:, 0].cpu().numpy())
            trues_volume.extend(y[:, 1].cpu().numpy())
            loss1 = criterion(price_pred.squeeze(), y[:, 0])
            loss2 = criterion(volume_pred.squeeze(), y[:, 1])
            total_loss += (loss1 + loss2).item()
    avg_loss = total_loss / len(test_loader)
    print(f"테스트셋 평균 손실: {avg_loss:.6f}")

    # 예측값/실제값 비교 저장
    df_compare = pd.DataFrame({
        'true_price': trues_price,
        'pred_price': preds_price,
        'true_volume': trues_volume,
        'pred_volume': preds_volume
    })
    compare_path = save_path.replace('.pth', '_test_compare.csv')
    df_compare.to_csv(compare_path, index=False)
    print(f"예측/실제값 비교 결과 저장: {compare_path}")

    # GPU/CPU 리소스 사용량 출력
    print(f"CPU 사용률: {psutil.cpu_percent()}%")
    print(f"메모리 사용률: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id} 사용률: {gpu.load*100:.1f}%, 메모리 사용률: {gpu.memoryUtil*100:.1f}%")


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
        # 데이터셋 분할: 0~30% 학습, 30~60% 검증, 60~100% 테스트
        train_df = df.iloc[:s1]
        val_df = df.iloc[s1:s2]
        test_df = df.iloc[s2:]

        train_ds = StockDataset(train_df, seq_len=SEQ_LEN)
        val_ds = StockDataset(val_df, seq_len=SEQ_LEN)
        test_ds = StockDataset(test_df, seq_len=SEQ_LEN)  # 필요시 사용

        tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        te_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)  # 필요시 사용

        model_path = os.path.join(MODEL_ROOT, f"{tk}.pth")
        # Pass required arguments to LSTMModel
        # train(LSTMModel(INPUT_SIZE, 5, 1, 1), tr_loader, va_loader, model_path)
        train(CustomLSTM(INPUT_SIZE, 5, SEQ_LEN), tr_loader, va_loader, model_path)
        print(f"✅ Training completed for {tk}. Model saved at {model_path}")


        # main 함수 마지막 부분에 아래 코드 추가 예시:
        evaluate(CustomLSTM(INPUT_SIZE, 5, SEQ_LEN), te_loader, model_path)

if __name__ == "__main__":
    main()
