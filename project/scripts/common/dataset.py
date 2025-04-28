# common/dataset.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_LEN = 60  # 입력 시퀀스 길이 (고정)

class StockDataset(Dataset):
    def __init__(self, dataframe):
        """
        Args:
            dataframe (pd.DataFrame): OHLCV 데이터 (Open, High, Low, Close, Volume)
        """
        self.X = []
        self.y = []
        data = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']].values

        for i in range(len(data) - SEQ_LEN):
            self.X.append(data[i:i+SEQ_LEN])
            self.y.append(data[i+SEQ_LEN][3])  # 다음 타임스텝 Close 가격 (index=3)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
