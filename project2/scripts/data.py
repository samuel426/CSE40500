import os
import yfinance as yf
import pandas as pd

# 1. 수집할 종목과 폴더 이름 매칭 (✅ S&P500 제거, ✅ KOSPI 추가)
tickers = {
    # "KOSPI": "^KS11",       # 코스피 종합지수
    "Apple": "AAPL",
    # "NASDAQ": "^IXIC",
    # "Tesla": "TSLA",
    # "Samsung": "005930.KS"
}

# 2. 저장할 루트 디렉토리
root_dir = "data"

# 3. 데이터 수집 및 저장
for folder_name, ticker_symbol in tickers.items():
    print(f"📥 Downloading data for {folder_name} ({ticker_symbol})...")
    
    # 3-1. 다운로드
    start_date = "2021-01-01"
    end_date = "2025-05-23"
    # yfinance를 사용하여 데이터 다운로드
    df = yf.download(
        ticker_symbol,
        start=start_date,
        end=end_date,
        progress=False
    )

    # 3-2. 필요한 열만 추출
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # 3-3. 결측치 처리 (선형 보간법)
    df.interpolate(method='linear', inplace=True)
    
    # 3-4. 저장 폴더 생성
    save_path = os.path.join(root_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)
    
    # 3-5. CSV로 저장
    file_path = os.path.join(save_path, f"{start_date}_{end_date}_ohlcv.csv")
    df.to_csv(file_path)
    
    print(f"✅ Saved to {file_path}")

print("🎯 모든 데이터 다운로드 및 저장 완료.")
