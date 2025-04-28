import yfinance as yf

# 예시: 삼성전자 (KS는 한국 거래소), AAPL은 애플
samsung = yf.Ticker("005930.KS")  # 삼성전자
apple = yf.Ticker("AAPL")        # 애플

# 최근 1개월 데이터
df = samsung.history(period="1mo")

# 특정 날짜 범위 데이터
df_range = apple.history(start="2023-01-01", end="2023-12-31")

print(df_range.head())
