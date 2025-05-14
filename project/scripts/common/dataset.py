# common/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """
    OHLCV -> NHWC([1, seq_len, 5])  +  target = Close(price) 1-scalar
    Height(H)=1, Width(W)=seq_len, Channels(C)=feature_dim
    """
    def __init__(self, dataframe, seq_len: int = 5):
        self.seq_len = seq_len
        data = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)

        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])       # (seq_len, 5)
            y.append(data[i + seq_len][3])      # Close price
        self.X = np.expand_dims(np.array(X, dtype=np.float32), axis=1)   # (N, 1, seq_len, 5)
        self.y = np.array(y, dtype=np.float32)                           # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])
