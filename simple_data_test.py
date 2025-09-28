# Simple script to download and explore Apple stock data
import yfinance as yf
import pandas as pd

print("=== Getting Apple Stock Data ===")

# Step 1: Create a "ticker" object for Apple
apple = yf.Ticker("AAPL")

# Step 2: Download the last 3 months of data
print("Downloading Apple data for the last 3 months...")
data = apple.history(period="3mo")

# Step 3: Look at what we got
print(f"\nWe downloaded {len(data)} days of data")
print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")

# Step 4: See the structure
print(f"\nColumns (what data we have): {list(data.columns)}")
print(f"Data shape (rows, columns): {data.shape}")

# Step 5: Look at the first few days
print("\n=== First 5 days ===")
print(data.head())

# Step 6: Look at the last few days  
print("\n=== Last 5 days ===")
print(data.tail())

# Step 7: Some basic statistics
print("\n=== Basic Statistics ===")
print(f"Lowest closing price: ${data['Close'].min():.2f}")
print(f"Highest closing price: ${data['Close'].max():.2f}")
print(f"Average closing price: ${data['Close'].mean():.2f}")
print(f"Latest closing price: ${data['Close'].iloc[-1]:.2f}")

# Step 8: Save it to a file so we can use it later
print("\n=== Saving Data ===")
data.to_csv("data/apple_3months.csv")
print("Saved to data/apple_3months.csv")

print("\n🎉 Success! We now have Apple stock data to work with!")