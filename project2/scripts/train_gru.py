import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common.dataset import StockDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 5  # 입력 특성 수 (Open, High, Low, Close, Volume)
SEQ_LEN = 8       # 시퀀스 길이
HIDDEN_SIZE = 5   # Hailo에서 in_channels=out_channels, 5 이어야 함
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

DATA_ROOT = "./data"
MODEL_ROOT = "./models/GRU"
TICKERS = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]
TICKERS = ["Apple"]

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.fc_price_conv = nn.Conv2d(HIDDEN_SIZE, 1, kernel_size=1)
        self.fc_volume_conv = nn.Conv2d(HIDDEN_SIZE, 1, kernel_size=1)

    def forward(self, x, h0=None):
        out, h = self.gru(x, h0)
        out = out[:, -1, :]  # (B, 5)
        out = out.unsqueeze(-1).unsqueeze(-1)  # (B, 5, 1, 1)
        price = self.fc_price_conv(out).view(out.size(0), -1)   # (B, 1)
        volume = self.fc_volume_conv(out).view(out.size(0), -1) # (B, 1)
        return price, volume

def train_org(model, train_loader, val_loader, save_path):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            h0 = torch.zeros(1, x.size(0), HIDDEN_SIZE).to(DEVICE)  # 초기 hidden state
            optimizer.zero_grad()
            price_pred, volume_pred = model(x, h0)
            loss1 = criterion(price_pred.squeeze(), y[:, 0])  # 가격 예측 손실
            loss2 = criterion(volume_pred.squeeze(), y[:, 1])  # 거래량 예측 손실
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
                h0 = torch.zeros(1, x.size(0), HIDDEN_SIZE).to(DEVICE)
                price_pred, volume_pred = model(x, h0)
                loss1 = criterion(price_pred.squeeze(), y[:, 0])
                loss2 = criterion(volume_pred.squeeze(), y[:, 1])
                loss = loss1 + loss2
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"[{epoch:02}/{EPOCHS}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # 최적의 모델 저장
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
            h0 = torch.zeros(1, x.size(0), HIDDEN_SIZE).to(DEVICE)
            optimizer.zero_grad()
            price_pred, volume_pred = model(x, h0)
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
                h0 = torch.zeros(1, x.size(0), HIDDEN_SIZE).to(DEVICE)
                price_pred, volume_pred = model(x, h0)
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
        h0 = torch.zeros(1, x.size(0), HIDDEN_SIZE).to(DEVICE)
        start = time.time()
        for _ in range(100):
            _ = model(x, h0)
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
    try:
        import GPUtil
        gpu_available = True
    except ImportError:
        gpu_available = False

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
            h0 = torch.zeros(1, x.size(0), HIDDEN_SIZE).to(DEVICE)
            price_pred, volume_pred = model(x, h0)
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
    if torch.cuda.is_available() and gpu_available:
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

        # 데이터 로드 및 전처리
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
        print(f"Training on {tk}: {len(train_df)} samples, Validation: {len(val_df)} samples, Test: {len(test_df)} samples")
        train_ds = StockDataset(train_df, seq_len=SEQ_LEN)
        val_ds = StockDataset(val_df, seq_len=SEQ_LEN)
        test_ds = StockDataset(test_df, seq_len=SEQ_LEN)  # 필요시 사용

        tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        te_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)  # 필요시 사용
        print(f"Data loaded for {tk}: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

        model_path = os.path.join(MODEL_ROOT, f"{tk}.pth")
        train(GRUModel(), tr_loader, va_loader, model_path)
        print(f"✅ Training completed for {tk}. Model saved at {model_path}")
        # 사용 예시 (main 함수 마지막에 추가):
        evaluate(GRUModel(), te_loader, model_path)


if __name__ == "__main__":
    main()