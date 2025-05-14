import torch

# 하이퍼 파라미터 모듈로 공유 (hyp.py 등으로 저장해도 되고 스크립트마다 copy)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE    = 5     # feature  dimension (C)
HIDDEN_SIZE   = 64
SEQ_LEN       = 5
BATCH_SIZE    = 32
EPOCHS        = 30
LR            = 1e-3
TICKERS       = ["KOSPI", "Apple", "NASDAQ", "Tesla", "Samsung"]
DATA_ROOT     = "./data"
