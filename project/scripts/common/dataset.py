# common/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    """
    OHLCV → [N, seq_len, 5]  /  target = [종가, 거래량]
    """
    def __init__(self, dataframe, seq_len=60):
        self.seq_len = seq_len
        data = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']].values
        self.X, self.y = [], []

        for i in range(len(data) - seq_len):
            # N, seq_len, input_size 형태로 맞추기
            self.X.append(data[i:i+seq_len])  # (seq_len, 5)
            self.y.append([data[i+seq_len][3], data[i+seq_len][4]])

        self.X = np.array(self.X, dtype=np.float32)  # (N, seq_len, 5)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx])  # (seq_len, 5)
        y = torch.tensor(self.y[idx], dtype=torch.float32)  # [종가, 거래량]
        return X, y
