import pandas as pd

df = pd.read_csv('./data/KOSPI/ohlcv.csv')
print(df.columns)
print(df.head())
