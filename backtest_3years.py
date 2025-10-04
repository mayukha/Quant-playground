import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuration
TICKER = "AAPL"
INITIAL_CAPITAL = 10000
COMMISSION = 1.0  # $ per trade
SLIPPAGE = 0.001  # 0.1% slippage
STOP_LOSS = 0.03  # 3% stop loss
TAKE_PROFIT = 0.08  # 8% take profit

# Strategy parameters
SHORT_MA = 10
LONG_MA = 20
VOLUME_MA = 5

print("=" * 60)
print("📈 3-YEAR MOVING AVERAGE CROSSOVER BACKTEST")
print("=" * 60)

# Step 1: Download 3 years of data
print("\n1️⃣ Downloading 3 years of data...")
ticker = yf.Ticker(TICKER)
df = ticker.history(period="3y")

# Save to CSV
Path("data").mkdir(exist_ok=True)
df.to_csv('data/apple_3years.csv')
print(f"✓ Downloaded {len(df)} days of data")
print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")

# Step 2: Calculate indicators
print("\n2️⃣ Calculating indicators...")
df['MA_short'] = df['Close'].rolling(window=SHORT_MA).mean()
df['MA_long'] = df['Close'].rolling(window=LONG_MA).mean()
df['Volume_MA'] = df['Volume'].rolling(window=VOLUME_MA).mean()
df['Returns'] = df['Close'].pct_change()

# Step 3: Generate signals
print("3️⃣ Generating trading signals...")
df['Signal'] = 0

# Buy signal: short MA crosses above long MA AND volume > average
df.loc[(df['MA_short'] > df['MA_long']) & 
       (df['MA_short'].shift(1) <= df['MA_long'].shift(1)) &
       (df['Volume'] > df['Volume_MA']), 'Signal'] = 1

# Sell signal: short MA crosses below long MA
df.loc[(df['MA_short'] < df['MA_long']) & 
       (df['MA_short'].shift(1) >= df['MA_long'].shift(1)), 'Signal'] = -1

df = df.dropna()

# Step 4: Backtest
print("4️⃣ Running backtest...")
position = 0
cash = INITIAL_CAPITAL
shares = 0
entry_price = 0
trades = []
portfolio_value = []

for i in range(len(df)):
    date = df.index[i]
    price = df['Close'].iloc[i]
    signal = df['Signal'].iloc[i]
    
    # Check stop-loss and take-profit
    if position == 1 and shares > 0:
        pnl_pct = (price - entry_price) / entry_price
        
        if pnl_pct <= -STOP_LOSS:  # Stop loss hit
            cash = shares * price * (1 - SLIPPAGE) - COMMISSION
            trades.append({
                'Date': date,
                'Type': 'SELL (Stop Loss)',
                'Price': price,
                'Shares': shares,
                'Cash': cash,
                'PnL%': pnl_pct * 100
            })
            position = 0
            shares = 0
            
        elif pnl_pct >= TAKE_PROFIT:  # Take profit hit
            cash = shares * price * (1 - SLIPPAGE) - COMMISSION
            trades.append({
                'Date': date,
                'Type': 'SELL (Take Profit)',
                'Price': price,
                'Shares': shares,
                'Cash': cash,
                'PnL%': pnl_pct * 100
            })
            position = 0
            shares = 0
    
    # Regular signals
    if signal == 1 and position == 0:  # Buy signal
        shares = (cash * (1 - SLIPPAGE) - COMMISSION) / price
        entry_price = price
        position = 1
        trades.append({
            'Date': date,
            'Type': 'BUY',
            'Price': price,
            'Shares': shares,
            'Cash': cash,
            'PnL%': 0
        })
        cash = 0
        
    elif signal == -1 and position == 1:  # Sell signal
        cash = shares * price * (1 - SLIPPAGE) - COMMISSION
        pnl_pct = (price - entry_price) / entry_price
        trades.append({
            'Date': date,
            'Type': 'SELL',
            'Price': price,
            'Shares': shares,
            'Cash': cash,
            'PnL%': pnl_pct * 100
        })
        position = 0
        shares = 0
    
    # Track portfolio value
    if position == 1:
        portfolio_value.append(shares * price)
    else:
        portfolio_value.append(cash)

# Final position closing
if position == 1:
    final_price = df['Close'].iloc[-1]
    cash = shares * final_price * (1 - SLIPPAGE) - COMMISSION
    pnl_pct = (final_price - entry_price) / entry_price
    trades.append({
        'Date': df.index[-1],
        'Type': 'SELL (Final)',
        'Price': final_price,
        'Shares': shares,
        'Cash': cash,
        'PnL%': pnl_pct * 100
    })

final_capital = cash if position == 0 else shares * df['Close'].iloc[-1]

# Step 5: Calculate metrics
print("\n5️⃣ Calculating performance metrics...")

trades_df = pd.DataFrame(trades)
df['Portfolio_Value'] = portfolio_value

# Strategy metrics
total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
num_trades = len(trades_df)
winning_trades = trades_df[trades_df['PnL%'] > 0]
win_rate = len(winning_trades) / (num_trades / 2) * 100 if num_trades > 0 else 0

# Calculate returns for Sharpe ratio
df['Strategy_Returns'] = df['Portfolio_Value'].pct_change()
sharpe_ratio = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(252)

# Maximum drawdown
cumulative = (1 + df['Strategy_Returns']).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min() * 100

# Buy and hold comparison
buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

# Annualized returns
years = len(df) / 252
annualized_return = (final_capital / INITIAL_CAPITAL) ** (1/years) - 1
annualized_return_pct = annualized_return * 100

# Step 6: Print results
print("\n" + "=" * 60)
print("📊 PERFORMANCE SUMMARY (3 YEARS)")
print("=" * 60)
print(f"\n💰 RETURNS:")
print(f"   Initial Capital:        ${INITIAL_CAPITAL:,.2f}")
print(f"   Final Capital:          ${final_capital:,.2f}")
print(f"   Total Return:           {total_return:+.2f}%")
print(f"   Annualized Return:      {annualized_return_pct:+.2f}%")
print(f"   Buy & Hold Return:      {buy_hold_return:+.2f}%")
print(f"   Outperformance:         {total_return - buy_hold_return:+.2f}%")

print(f"\n📈 RISK METRICS:")
print(f"   Sharpe Ratio:           {sharpe_ratio:.2f}")
print(f"   Maximum Drawdown:       {max_drawdown:.2f}%")
print(f"   Volatility (Annual):    {df['Strategy_Returns'].std() * np.sqrt(252) * 100:.2f}%")

print(f"\n🎯 TRADING STATISTICS:")
print(f"   Total Trades:           {num_trades}")
print(f"   Winning Trades:         {len(winning_trades)}")
print(f"   Win Rate:               {win_rate:.1f}%")
print(f"   Avg Trade Duration:     {len(df) / (num_trades/2):.1f} days" if num_trades > 0 else "   Avg Trade Duration:     N/A")
print(f"   Trades per Year:        {(num_trades/2) / years:.1f}")

if len(winning_trades) > 0:
    losing_trades = trades_df[trades_df['PnL%'] < 0]
    avg_win = winning_trades['PnL%'].mean()
    avg_loss = losing_trades['PnL%'].mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades['PnL%'].sum() / losing_trades['PnL%'].sum()) if len(losing_trades) > 0 else float('inf')
    
    print(f"   Average Win:            {avg_win:.2f}%")
    print(f"   Average Loss:           {avg_loss:.2f}%")
    print(f"   Profit Factor:          {profit_factor:.2f}")

print("\n" + "=" * 60)

# Step 7: Visualizations
print("\n6️⃣ Creating visualizations...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle(f'{TICKER} - 3 Year MA Crossover Backtest Results', fontsize=16, fontweight='bold')

# Chart 1: Price and signals
ax1 = axes[0]
ax1.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1.5, alpha=0.7)
ax1.plot(df.index, df['MA_short'], label=f'{SHORT_MA}-day MA', color='blue', linewidth=1, alpha=0.7)
ax1.plot(df.index, df['MA_long'], label=f'{LONG_MA}-day MA', color='red', linewidth=1, alpha=0.7)

# Mark trades
buy_signals = trades_df[trades_df['Type'] == 'BUY']
sell_signals = trades_df[trades_df['Type'].str.contains('SELL')]

ax1.scatter(buy_signals['Date'], buy_signals['Price'], color='green', marker='^', s=100, label='Buy', zorder=5)
ax1.scatter(sell_signals['Date'], sell_signals['Price'], color='red', marker='v', s=100, label='Sell', zorder=5)

ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
ax1.set_title('Price Chart with Trading Signals', fontsize=12, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Chart 2: Portfolio value comparison
ax2 = axes[1]
ax2.plot(df.index, df['Portfolio_Value'], label='Strategy Portfolio', color='green', linewidth=2)

# Buy and hold portfolio
buy_hold_portfolio = INITIAL_CAPITAL * (df['Close'] / df['Close'].iloc[0])
ax2.plot(df.index, buy_hold_portfolio, label='Buy & Hold', color='blue', linewidth=2, linestyle='--')

ax2.axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', label='Initial Capital')
ax2.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
ax2.set_title('Portfolio Value: Strategy vs Buy & Hold', fontsize=12, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Chart 3: Drawdown
ax3 = axes[2]
ax3.fill_between(df.index, drawdown * 100, 0, color='red', alpha=0.3)
ax3.plot(df.index, drawdown * 100, color='darkred', linewidth=1)
ax3.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
ax3.set_title(f'Strategy Drawdown (Max: {max_drawdown:.2f}%)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/backtest_3years_results.png', dpi=300, bbox_inches='tight')
print("✓ Charts saved to: data/backtest_3years_results.png")

# Save trades to CSV
trades_df.to_csv('data/trades_3years.csv', index=False)
print("✓ Trade log saved to: data/trades_3years.csv")

print("\n✅ 3-year backtest complete!")
print("\nNext steps:")
print("  • Review the charts in data/backtest_3years_results.png")
print("  • Check trade details in data/trades_3years.csv")
print("  • Compare with your 3-month results")
print("  • Ready to test 10 years? Let me know!")