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
print("📈 10-YEAR MOVING AVERAGE CROSSOVER BACKTEST")
print("=" * 60)

# Step 1: Download 10 years of data
print("\n1️⃣ Downloading 10 years of data...")
ticker = yf.Ticker(TICKER)
df = ticker.history(period="10y")

# Save to CSV
Path("data").mkdir(exist_ok=True)
df.to_csv('data/apple_10years.csv')
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

buy_hold_annualized = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (1/years) - 1) * 100

# Calculate year-by-year performance
df['Year'] = df.index.year
yearly_performance = []

for year in sorted(df['Year'].unique()):
    year_data = df[df['Year'] == year]
    if len(year_data) > 0:
        year_start_val = year_data['Portfolio_Value'].iloc[0]
        year_end_val = year_data['Portfolio_Value'].iloc[-1]
        year_return = (year_end_val - year_start_val) / year_start_val * 100
        
        year_start_price = year_data['Close'].iloc[0]
        year_end_price = year_data['Close'].iloc[-1]
        bh_return = (year_end_price - year_start_price) / year_start_price * 100
        
        yearly_performance.append({
            'Year': year,
            'Strategy Return': year_return,
            'Buy & Hold Return': bh_return,
            'Outperformance': year_return - bh_return
        })

yearly_df = pd.DataFrame(yearly_performance)

# Step 6: Print results
print("\n" + "=" * 60)
print("📊 PERFORMANCE SUMMARY (10 YEARS)")
print("=" * 60)
print(f"\n💰 RETURNS:")
print(f"   Initial Capital:        ${INITIAL_CAPITAL:,.2f}")
print(f"   Final Capital:          ${final_capital:,.2f}")
print(f"   Total Return:           {total_return:+.2f}%")
print(f"   Annualized Return:      {annualized_return_pct:+.2f}%")
print(f"   Buy & Hold Return:      {buy_hold_return:+.2f}%")
print(f"   BH Annualized:          {buy_hold_annualized:+.2f}%")
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

print("\n📅 YEAR-BY-YEAR PERFORMANCE:")
print("-" * 60)
for _, row in yearly_df.iterrows():
    symbol = "🟢" if row['Outperformance'] > 0 else "🔴"
    print(f"   {symbol} {int(row['Year'])}: Strategy {row['Strategy Return']:+6.2f}% | "
          f"B&H {row['Buy & Hold Return']:+6.2f}% | Diff {row['Outperformance']:+6.2f}%")

print("\n" + "=" * 60)

# Step 7: Visualizations
print("\n6️⃣ Creating visualizations...")

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Chart 1: Price and signals (large, spans both columns)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1.5, alpha=0.7)
ax1.plot(df.index, df['MA_short'], label=f'{SHORT_MA}-day MA', color='blue', linewidth=1, alpha=0.7)
ax1.plot(df.index, df['MA_long'], label=f'{LONG_MA}-day MA', color='red', linewidth=1, alpha=0.7)

# Mark trades (sample to avoid overcrowding)
buy_signals = trades_df[trades_df['Type'] == 'BUY']
sell_signals = trades_df[trades_df['Type'].str.contains('SELL')]

ax1.scatter(buy_signals['Date'], buy_signals['Price'], color='green', marker='^', s=50, label='Buy', zorder=5, alpha=0.6)
ax1.scatter(sell_signals['Date'], sell_signals['Price'], color='red', marker='v', s=50, label='Sell', zorder=5, alpha=0.6)

# Mark major events
ax1.axvline(pd.Timestamp('2020-03-01'), color='purple', linestyle='--', alpha=0.5, linewidth=2, label='COVID Crash')
ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
ax1.set_title('10-Year Price Chart with Trading Signals', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Chart 2: Portfolio value comparison
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(df.index, df['Portfolio_Value'], label='Strategy Portfolio', color='green', linewidth=2)

buy_hold_portfolio = INITIAL_CAPITAL * (df['Close'] / df['Close'].iloc[0])
ax2.plot(df.index, buy_hold_portfolio, label='Buy & Hold', color='blue', linewidth=2, linestyle='--')

ax2.axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', label='Initial Capital', alpha=0.5)
ax2.axvline(pd.Timestamp('2020-03-01'), color='purple', linestyle='--', alpha=0.3, linewidth=1)
ax2.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
ax2.set_title('Portfolio Value: Strategy vs Buy & Hold (10 Years)', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Chart 3: Drawdown
ax3 = fig.add_subplot(gs[2, :])
ax3.fill_between(df.index, drawdown * 100, 0, color='red', alpha=0.3)
ax3.plot(df.index, drawdown * 100, color='darkred', linewidth=1)
ax3.axvline(pd.Timestamp('2020-03-01'), color='purple', linestyle='--', alpha=0.3, linewidth=1)
ax3.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
ax3.set_title(f'Strategy Drawdown (Max: {max_drawdown:.2f}%)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Chart 4: Year-by-year comparison (bar chart)
ax4 = fig.add_subplot(gs[3, 0])
x = np.arange(len(yearly_df))
width = 0.35
bars1 = ax4.bar(x - width/2, yearly_df['Strategy Return'], width, label='Strategy', color='green', alpha=0.7)
bars2 = ax4.bar(x + width/2, yearly_df['Buy & Hold Return'], width, label='Buy & Hold', color='blue', alpha=0.7)
ax4.set_xlabel('Year', fontsize=10, fontweight='bold')
ax4.set_ylabel('Annual Return (%)', fontsize=10, fontweight='bold')
ax4.set_title('Year-by-Year Performance Comparison', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(yearly_df['Year'].astype(int), rotation=45)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Chart 5: Trade distribution
ax5 = fig.add_subplot(gs[3, 1])
sell_trades = trades_df[trades_df['Type'].str.contains('SELL') & (trades_df['PnL%'] != 0)]
if len(sell_trades) > 0:
    ax5.hist(sell_trades['PnL%'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax5.set_xlabel('Trade P&L (%)', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax5.set_title('Distribution of Trade Returns', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

fig.suptitle(f'{TICKER} - 10 Year MA Crossover Backtest Results', fontsize=16, fontweight='bold', y=0.995)

plt.savefig('data/backtest_10years_results.png', dpi=300, bbox_inches='tight')
print("✓ Charts saved to: data/backtest_10years_results.png")

# Save trades and yearly performance to CSV
trades_df.to_csv('data/trades_10years.csv', index=False)
yearly_df.to_csv('data/yearly_performance_10years.csv', index=False)
print("✓ Trade log saved to: data/trades_10years.csv")
print("✓ Yearly performance saved to: data/yearly_performance_10years.csv")

print("\n✅ 10-year backtest complete!")
print("\n🔍 KEY INSIGHTS:")
print(f"   • Strategy survived {years:.1f} years of trading")
print(f"   • Experienced the 2020 COVID crash")
print(f"   • Total trades executed: {num_trades}")
print(f"   • {'Beat' if total_return > buy_hold_return else 'Underperformed'} buy-and-hold by {abs(total_return - buy_hold_return):.2f}%")

if max_drawdown < -20:
    print(f"   ⚠️  Large drawdown detected: {max_drawdown:.2f}%")
else:
    print(f"   ✓ Controlled drawdown: {max_drawdown:.2f}%")

print("\nNext steps:")
print("  • Review the 10-year charts in data/backtest_10years_results.png")
print("  • Compare with your 3-year results")
print("  • Analyze year-by-year performance patterns")
print("  • Ready to optimize the strategy? Let me know!")