import yfinance as yf
import pandas as pd
from pathlib import Path

# Ensure data directory exists
Path("data").mkdir(exist_ok=True)

print("Downloading Apple data...")

# Download 3 years
ticker = yf.Ticker("AAPL")
df_3years = ticker.history(period="3y")
df_3years.to_csv('data/apple_3years.csv')
print(f"✓ 3-year data: {len(df_3years)} days saved")

# Download 10 years
df_10years = ticker.history(period="10y")
df_10years.to_csv('data/apple_10years.csv')
print(f"✓ 10-year data: {len(df_10years)} days saved")

print("\nData ready for backtesting!")
