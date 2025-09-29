# Let's understand our Apple stock data in detail
import pandas as pd
import numpy as np

print("=== Loading and Understanding Apple Stock Data ===\n")

# Load the data we saved earlier
data = pd.read_csv("data/apple_3months.csv", index_col=0, parse_dates=True)

print(f"📊 Data Overview:")
print(f"   • Total days: {len(data)}")
print(f"   • Date range: {data.index[0].date()} to {data.index[-1].date()}")
print(f"   • Columns: {list(data.columns)}")
print(f"   • Data shape: {data.shape} (rows, columns)")

# Let's understand each column
print(f"\n🔍 What Each Column Means:")
print(f"   • Open: First price when market opened that day")
print(f"   • High: Highest price during the day")
print(f"   • Low: Lowest price during the day") 
print(f"   • Close: Last price when market closed (most important!)")
print(f"   • Volume: Number of shares traded that day")
print(f"   • Dividends: Money paid to shareholders (usually $0 most days)")
print(f"   • Stock Splits: When 1 share becomes 2+ shares (rare)")

# Look at a specific day in detail
print(f"\n📅 Example: Let's look at one specific day")
sample_day = data.iloc[30]  # Pick the 30th day
sample_date = data.index[30].date()
print(f"   Date: {sample_date}")
print(f"   • Market opened at: ${sample_day['Open']:.2f}")
print(f"   • Highest price: ${sample_day['High']:.2f}")
print(f"   • Lowest price: ${sample_day['Low']:.2f}")
print(f"   • Market closed at: ${sample_day['Close']:.2f}")
print(f"   • Volume traded: {sample_day['Volume']:,.0f} shares")
print(f"   • Daily price range: ${sample_day['High'] - sample_day['Low']:.2f}")

# Calculate some interesting statistics
print(f"\n📈 Price Statistics Over 3 Months:")
close_prices = data['Close']
print(f"   • Lowest closing price: ${close_prices.min():.2f}")
print(f"   • Highest closing price: ${close_prices.max():.2f}")
print(f"   • Average closing price: ${close_prices.mean():.2f}")
print(f"   • Current price: ${close_prices.iloc[-1]:.2f}")
print(f"   • Total price change: ${close_prices.iloc[-1] - close_prices.iloc[0]:.2f}")
print(f"   • Percentage change: {((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) * 100:.1f}%")

# Look at daily price movements
print(f"\n📊 Daily Price Movements:")
data['Daily_Change'] = data['Close'] - data['Open']
data['Daily_Change_Pct'] = ((data['Close'] / data['Open']) - 1) * 100

positive_days = (data['Daily_Change'] > 0).sum()
negative_days = (data['Daily_Change'] < 0).sum()
flat_days = (data['Daily_Change'] == 0).sum()

print(f"   • Days stock went UP: {positive_days} ({positive_days/len(data)*100:.1f}%)")
print(f"   • Days stock went DOWN: {negative_days} ({negative_days/len(data)*100:.1f}%)")
print(f"   • Days stock stayed FLAT: {flat_days} ({flat_days/len(data)*100:.1f}%)")
print(f"   • Biggest daily gain: ${data['Daily_Change'].max():.2f} ({data['Daily_Change_Pct'].max():.2f}%)")
print(f"   • Biggest daily loss: ${data['Daily_Change'].min():.2f} ({data['Daily_Change_Pct'].min():.2f}%)")
print(f"   • Average daily change: ${data['Daily_Change'].mean():.2f}")

# Look at volume patterns
print(f"\n📦 Trading Volume Insights:")
avg_volume = data['Volume'].mean()
print(f"   • Average daily volume: {avg_volume:,.0f} shares")
print(f"   • Highest volume day: {data['Volume'].max():,.0f} shares")
print(f"   • Lowest volume day: {data['Volume'].min():,.0f} shares")

# Find the most volatile days
print(f"\n🎢 Most Volatile Days (biggest price swings):")
data['Daily_Range'] = data['High'] - data['Low']
data['Daily_Range_Pct'] = (data['Daily_Range'] / data['Open']) * 100

most_volatile = data.nlargest(3, 'Daily_Range_Pct')
for i, (date, row) in enumerate(most_volatile.iterrows(), 1):
    print(f"   {i}. {date.date()}: ${row['Daily_Range']:.2f} swing ({row['Daily_Range_Pct']:.2f}% of opening price)")

# Show first and last few days for comparison
print(f"\n📋 First 5 Days vs Last 5 Days:")
print("\nFIRST 5 DAYS:")
print(data[['Open', 'High', 'Low', 'Close', 'Volume']].head().round(2))
print("\nLAST 5 DAYS:")
print(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail().round(2))

print(f"\n✅ Data exploration complete! Now you understand what we're working with.")