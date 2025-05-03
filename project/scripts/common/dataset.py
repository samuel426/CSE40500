# common/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    """
    OHLCV → [N‑seq_len, seq_len, 5]  /  target = 종가(다음 시점)
    """
    def __init__(self, dataframe, seq_len=60):
        self.seq_len = seq_len

        data = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']].values
        self.X, self.y = [], []

        # 마지막 시점의 Close 를 예측 대상으로
        for i in range(len(data) - seq_len):
            self.X.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len][3])  # Close 열

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)  # 이 부분 수정
        return X, y

