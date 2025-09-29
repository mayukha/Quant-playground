import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

# Load your Apple data
df = pd.read_csv('data/apple_3months.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Calculate additional metrics for deeper analysis
df['Daily_Return'] = df['Close'].pct_change() * 100  # Percentage return
df['Price_Range'] = df['High'] - df['Low']  # Daily range
df['Body_Size'] = abs(df['Close'] - df['Open'])  # Candlestick body
df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)  # Upper wick
df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']  # Lower wick
df['Volume_MA'] = df['Volume'].rolling(window=5).mean()  # 5-day volume average
df['Price_MA_10'] = df['Close'].rolling(window=10).mean()  # 10-day moving average
df['Price_MA_20'] = df['Close'].rolling(window=20).mean()  # 20-day moving average

# Create comprehensive visualization dashboard
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 24))

# 1. MAIN PRICE CHART with Moving Averages
ax1 = plt.subplot(4, 2, (1, 2))  # Top row, spanning 2 columns
ax1.plot(df['Date'], df['Close'], linewidth=2, label='Close Price', color='#1f77b4')
ax1.plot(df['Date'], df['Price_MA_10'], linewidth=1.5, label='10-day MA', color='orange', alpha=0.8)
ax1.plot(df['Date'], df['Price_MA_20'], linewidth=1.5, label='20-day MA', color='red', alpha=0.8)
ax1.fill_between(df['Date'], df['Low'], df['High'], alpha=0.1, color='gray', label='Daily Range')
ax1.set_title('AAPL Stock Price with Moving Averages (3 Months)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
# Format x-axis dates
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 2. DAILY RETURNS ANALYSIS
ax2 = plt.subplot(4, 2, 3)
colors = ['green' if x > 0 else 'red' for x in df['Daily_Return'].dropna()]
ax2.bar(range(len(df['Daily_Return'].dropna())), df['Daily_Return'].dropna(), 
        color=colors, alpha=0.7, width=0.8)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_title('Daily Returns (%)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Return (%)', fontsize=12)
ax2.set_xlabel('Trading Days', fontsize=12)
ax2.grid(True, alpha=0.3)

# Add statistics text
returns = df['Daily_Return'].dropna()
stats_text = f'Avg: {returns.mean():.2f}%\nStd: {returns.std():.2f}%\nMax: {returns.max():.2f}%\nMin: {returns.min():.2f}%'
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

# 3. VOLUME ANALYSIS
ax3 = plt.subplot(4, 2, 4)
ax3.bar(df['Date'], df['Volume']/1e6, alpha=0.6, color='purple', width=1)
ax3.plot(df['Date'], df['Volume_MA']/1e6, color='darkred', linewidth=2, label='5-day MA')
ax3.set_title('Volume Analysis (Millions)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Volume (Millions)', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# 4. CANDLESTICK PATTERNS ANALYSIS
ax4 = plt.subplot(4, 2, 5)
# Histogram of body sizes
ax4.hist(df['Body_Size'], bins=15, alpha=0.7, color='blue', edgecolor='black')
ax4.set_title('Distribution of Candlestick Body Sizes', fontsize=14, fontweight='bold')
ax4.set_xlabel('Body Size ($)', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.grid(True, alpha=0.3)

# Add statistics
body_stats = f'Mean: ${df["Body_Size"].mean():.2f}\nMedian: ${df["Body_Size"].median():.2f}'
ax4.text(0.98, 0.98, body_stats, transform=ax4.transAxes, verticalalignment='top', 
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 5. VOLATILITY PATTERNS
ax5 = plt.subplot(4, 2, 6)
df['Volatility'] = df['Price_Range'] / df['Close'] * 100  # Volatility as % of close price
ax5.plot(df['Date'], df['Volatility'], color='red', linewidth=1.5, alpha=0.8)
ax5.fill_between(df['Date'], df['Volatility'], alpha=0.3, color='red')
ax5.set_title('Daily Volatility (High-Low as % of Close)', fontsize=14, fontweight='bold')
ax5.set_ylabel('Volatility (%)', fontsize=12)
ax5.grid(True, alpha=0.3)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

# Add volatility statistics
vol_avg = df['Volatility'].mean()
ax5.axhline(y=vol_avg, color='black', linestyle='--', alpha=0.7, label=f'Avg: {vol_avg:.2f}%')
ax5.legend()

# 6. CORRELATION HEATMAP
ax6 = plt.subplot(4, 2, 7)
correlation_data = df[['Close', 'Volume', 'Daily_Return', 'Price_Range', 'Body_Size']].corr()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax6)
ax6.set_title('Correlation Matrix', fontsize=14, fontweight='bold')

# 7. ADVANCED PATTERN: Price vs Volume Relationship
ax7 = plt.subplot(4, 2, 8)
# Color points by daily return
scatter = ax7.scatter(df['Volume']/1e6, df['Close'], 
                     c=df['Daily_Return'], cmap='RdYlGn', 
                     alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax7.set_xlabel('Volume (Millions)', fontsize=12)
ax7.set_ylabel('Close Price ($)', fontsize=12)
ax7.set_title('Price vs Volume (colored by daily return)', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax7)
cbar.set_label('Daily Return (%)', rotation=270, labelpad=15)

plt.tight_layout(pad=3.0)
plt.savefig('data/apple_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# DETAILED INSIGHTS ANALYSIS
print("=" * 80)
print("🔍 DETAILED STOCK ANALYSIS INSIGHTS")
print("=" * 80)

# 1. Price Movement Insights
price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
print(f"\n📈 OVERALL PERFORMANCE:")
print(f"   • Total return over period: {price_change:.2f}%")
print(f"   • Starting price: ${df['Close'].iloc[0]:.2f}")
print(f"   • Ending price: ${df['Close'].iloc[-1]:.2f}")

# 2. Volatility Analysis
avg_volatility = df['Volatility'].mean()
high_vol_days = len(df[df['Volatility'] > avg_volatility * 1.5])
print(f"\n📊 VOLATILITY INSIGHTS:")
print(f"   • Average daily volatility: {avg_volatility:.2f}%")
print(f"   • High volatility days (>1.5x avg): {high_vol_days}")
print(f"   • Most volatile day: {df.loc[df['Volatility'].idxmax(), 'Date'].strftime('%Y-%m-%d')} ({df['Volatility'].max():.2f}%)")

# 3. Volume Patterns
avg_volume = df['Volume'].mean()
high_volume_days = len(df[df['Volume'] > avg_volume * 1.2])
print(f"\n📉 VOLUME PATTERNS:")
print(f"   • Average daily volume: {avg_volume/1e6:.1f}M shares")
print(f"   • High volume days (>1.2x avg): {high_volume_days}")
print(f"   • Highest volume day: {df.loc[df['Volume'].idxmax(), 'Date'].strftime('%Y-%m-%d')} ({df['Volume'].max()/1e6:.1f}M)")

# 4. Moving Average Analysis
df['MA_Signal'] = np.where(df['Price_MA_10'] > df['Price_MA_20'], 'BUY', 'SELL')
current_signal = df['MA_Signal'].iloc[-1]
signal_changes = len(df[df['MA_Signal'] != df['MA_Signal'].shift(1)].dropna())
print(f"\n🎯 MOVING AVERAGE SIGNALS:")
print(f"   • Current signal (10 vs 20 MA): {current_signal}")
print(f"   • Signal changes during period: {signal_changes}")

# 5. Risk Metrics
returns = df['Daily_Return'].dropna()
sharpe_approx = returns.mean() / returns.std() if returns.std() > 0 else 0
down_days = len(returns[returns < 0])
up_days = len(returns[returns > 0])
print(f"\n⚠️  RISK METRICS:")
print(f"   • Daily Sharpe approximation: {sharpe_approx:.2f}")
print(f"   • Win rate: {up_days/(up_days+down_days)*100:.1f}% ({up_days}/{up_days+down_days} days)")
print(f"   • Largest single day loss: {returns.min():.2f}%")
print(f"   • Largest single day gain: {returns.max():.2f}%")

# 6. Pattern Recognition
big_moves = df[abs(df['Daily_Return']) > 2.0]  # Days with >2% moves
print(f"\n🔥 BIG MOVE ANALYSIS:")
print(f"   • Days with >2% moves: {len(big_moves)}")
if len(big_moves) > 0:
    print(f"   • Average volume on big move days: {big_moves['Volume'].mean()/1e6:.1f}M")
    print(f"   • Average volatility on big move days: {big_moves['Volatility'].mean():.2f}%")

print("\n" + "=" * 80)
print("📊 Chart saved as 'data/apple_comprehensive_analysis.png'")
print("=" * 80)