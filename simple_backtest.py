import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load your Apple data
print("=" * 80)
print("📊 SIMPLE BACKTESTING SYSTEM - COMPLETE EXAMPLE")
print("=" * 80)

df = pd.read_csv('data/apple_3months.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n✅ Loaded {len(df)} days of AAPL data")
print(f"   Period: {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")

# ============================================================================
# STEP 1: CALCULATE INDICATORS (Using only past data)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: CALCULATING INDICATORS")
print("=" * 80)

df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['Volume_MA'] = df['Volume'].rolling(window=5).mean()

# Important: First 20 rows have NaN for MA_20 (need 20 days of data to calculate)
print(f"✅ Calculated 10-day and 20-day moving averages")
print(f"   Note: First 20 days have NaN (insufficient data for calculation)")

# ============================================================================
# STEP 2: GENERATE SIGNALS (Buy/Sell rules)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: GENERATING TRADING SIGNALS")
print("=" * 80)

# Initialize signal column
df['Signal'] = 0  # 0 = do nothing, 1 = buy, -1 = sell

# Strategy Rules:
# BUY: When 10-day MA crosses ABOVE 20-day MA (and we have volume confirmation)
# SELL: When 10-day MA crosses BELOW 20-day MA

for i in range(1, len(df)):
    # Check if we have valid data (no NaN values)
    if pd.notna(df.loc[i, 'MA_10']) and pd.notna(df.loc[i, 'MA_20']):
        
        # Previous day's MAs
        prev_ma10 = df.loc[i-1, 'MA_10']
        prev_ma20 = df.loc[i-1, 'MA_20']
        
        # Today's MAs
        curr_ma10 = df.loc[i, 'MA_10']
        curr_ma20 = df.loc[i, 'MA_20']
        
        # BULLISH CROSSOVER: MA_10 crosses above MA_20
        if prev_ma10 <= prev_ma20 and curr_ma10 > curr_ma20:
            # Additional filter: Volume should be above average
            if df.loc[i, 'Volume'] > df.loc[i, 'Volume_MA']:
                df.loc[i, 'Signal'] = 1  # BUY signal
        
        # BEARISH CROSSOVER: MA_10 crosses below MA_20
        elif prev_ma10 >= prev_ma20 and curr_ma10 < curr_ma20:
            df.loc[i, 'Signal'] = -1  # SELL signal

buy_signals = len(df[df['Signal'] == 1])
sell_signals = len(df[df['Signal'] == -1])
print(f"✅ Generated signals:")
print(f"   Buy signals: {buy_signals}")
print(f"   Sell signals: {sell_signals}")

# ============================================================================
# STEP 3: BACKTEST ENGINE (Simulate trading)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: RUNNING BACKTEST SIMULATION")
print("=" * 80)

# Portfolio setup
initial_capital = 10000.0
cash = initial_capital
shares = 0
position_open = False
trade_log = []
portfolio_value = []

# Transaction costs
commission = 1.0  # $1 per trade (conservative estimate)
slippage_pct = 0.001  # 0.1% slippage (price moves against you)

# Risk management
stop_loss_pct = 0.03  # 3% stop loss
take_profit_pct = 0.08  # 8% take profit
entry_price = 0

# Simulate each trading day
for i in range(len(df)):
    current_date = df.loc[i, 'Date']
    current_price = df.loc[i, 'Close']
    signal = df.loc[i, 'Signal']
    
    # Check if we need to exit due to stop-loss or take-profit
    if position_open and shares > 0:
        # Calculate current profit/loss
        pct_change = (current_price - entry_price) / entry_price
        
        # STOP LOSS triggered
        if pct_change <= -stop_loss_pct:
            # Sell with slippage
            sell_price = current_price * (1 - slippage_pct)
            cash = shares * sell_price - commission
            profit = cash - (shares * entry_price)
            
            trade_log.append({
                'Date': current_date,
                'Action': 'SELL (Stop Loss)',
                'Price': sell_price,
                'Shares': shares,
                'Cash': cash,
                'Profit': profit,
                'Return_Pct': pct_change * 100
            })
            
            shares = 0
            position_open = False
        
        # TAKE PROFIT triggered
        elif pct_change >= take_profit_pct:
            # Sell with slippage
            sell_price = current_price * (1 - slippage_pct)
            cash = shares * sell_price - commission
            profit = cash - (shares * entry_price)
            
            trade_log.append({
                'Date': current_date,
                'Action': 'SELL (Take Profit)',
                'Price': sell_price,
                'Shares': shares,
                'Cash': cash,
                'Profit': profit,
                'Return_Pct': pct_change * 100
            })
            
            shares = 0
            position_open = False
    
    # Process BUY signal
    if signal == 1 and not position_open and cash > 0:
        # Buy with slippage (price moves against you)
        buy_price = current_price * (1 + slippage_pct)
        shares = (cash - commission) // buy_price  # How many shares can we afford?
        
        if shares > 0:
            cost = shares * buy_price + commission
            cash -= cost
            entry_price = buy_price
            position_open = True
            
            trade_log.append({
                'Date': current_date,
                'Action': 'BUY',
                'Price': buy_price,
                'Shares': shares,
                'Cash': cash,
                'Profit': 0,
                'Return_Pct': 0
            })
    
    # Process SELL signal (from strategy, not risk management)
    elif signal == -1 and position_open and shares > 0:
        # Sell with slippage
        sell_price = current_price * (1 - slippage_pct)
        cash = shares * sell_price - commission
        profit = cash - (shares * entry_price)
        pct_change = (sell_price - entry_price) / entry_price
        
        trade_log.append({
            'Date': current_date,
            'Action': 'SELL (Signal)',
            'Price': sell_price,
            'Shares': shares,
            'Cash': cash,
            'Profit': profit,
            'Return_Pct': pct_change * 100
        })
        
        shares = 0
        position_open = False
    
    # Calculate portfolio value for this day
    current_portfolio_value = cash + (shares * current_price if shares > 0 else 0)
    portfolio_value.append(current_portfolio_value)

df['Portfolio_Value'] = portfolio_value

print(f"✅ Backtest complete!")
print(f"   Total trades executed: {len(trade_log)}")

print("\n🔍 DEBUG INFORMATION:")
print(f"   Signals detected: {len(df[df['Signal'] != 0])}")
print(f"   Buy signals: {len(df[df['Signal'] == 1])}")
print(f"   Sell signals: {len(df[df['Signal'] == -1])}")
print(f"   Days with valid MA data: {len(df.dropna(subset=['MA_10', 'MA_20']))}")
print(f"   Cash remaining: ${cash:.2f}")
print(f"   Shares held: {shares}")
print(f"   Position open at end: {position_open}")

if len(trade_log) == 0:
    print("\n⚠️  NO TRADES EXECUTED - Possible reasons:")
    print("   1. No MA crossovers occurred during this period")
    print("   2. Volume filter blocked all signals")
    print("   3. Not enough data for MA calculation")

# ============================================================================
# STEP 4: CALCULATE PERFORMANCE METRICS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: PERFORMANCE ANALYSIS")
print("=" * 80)

# Final portfolio value
final_value = portfolio_value[-1]
total_return = ((final_value - initial_capital) / initial_capital) * 100

# Buy and hold comparison
buy_hold_shares = initial_capital / df.loc[0, 'Close']
buy_hold_value = buy_hold_shares * df.loc[len(df)-1, 'Close']
buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100

# Calculate returns for Sharpe ratio
df['Strategy_Return'] = df['Portfolio_Value'].pct_change()
mean_return = df['Strategy_Return'].mean()
std_return = df['Strategy_Return'].std()
sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0  # Annualized

# Maximum drawdown
df['Cumulative_Max'] = df['Portfolio_Value'].cummax()
df['Drawdown'] = (df['Portfolio_Value'] - df['Cumulative_Max']) / df['Cumulative_Max'] * 100
max_drawdown = df['Drawdown'].min()

# Trade statistics
trade_df = pd.DataFrame(trade_log)
if len(trade_df) > 0:
    profitable_trades = trade_df[trade_df['Profit'] > 0]
    losing_trades = trade_df[trade_df['Profit'] < 0]
    
    win_rate = len(profitable_trades) / len(trade_df[trade_df['Action'].str.contains('SELL')]) * 100 if len(trade_df[trade_df['Action'].str.contains('SELL')]) > 0 else 0
    avg_win = profitable_trades['Profit'].mean() if len(profitable_trades) > 0 else 0
    avg_loss = losing_trades['Profit'].mean() if len(losing_trades) > 0 else 0
    
    total_wins = profitable_trades['Profit'].sum() if len(profitable_trades) > 0 else 0
    total_losses = abs(losing_trades['Profit'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
else:
    win_rate = 0
    avg_win = 0
    avg_loss = 0
    profit_factor = 0

# Print results
print(f"\n💰 STRATEGY PERFORMANCE:")
print(f"   Initial Capital: ${initial_capital:,.2f}")
print(f"   Final Value: ${final_value:,.2f}")
print(f"   Total Return: {total_return:+.2f}%")
print(f"   ")
print(f"📊 BENCHMARK COMPARISON:")
print(f"   Buy & Hold Return: {buy_hold_return:+.2f}%")
print(f"   Strategy vs Buy & Hold: {total_return - buy_hold_return:+.2f}%")
print(f"   ")
print(f"📈 RISK METRICS:")
print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"   Maximum Drawdown: {max_drawdown:.2f}%")
print(f"   ")
print(f"🎯 TRADE STATISTICS:")
print(f"   Total Trades: {len(trade_df)}")
print(f"   Win Rate: {win_rate:.1f}%")
print(f"   Average Win: ${avg_win:.2f}")
print(f"   Average Loss: ${avg_loss:.2f}")
print(f"   Profit Factor: {profit_factor:.2f}")

# ============================================================================
# STEP 5: VISUALIZE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Chart 1: Price with Moving Averages and Trade Signals
ax1 = axes[0]
ax1.plot(df['Date'], df['Close'], label='Close Price', linewidth=2, color='black', alpha=0.7)
ax1.plot(df['Date'], df['MA_10'], label='10-day MA', linewidth=1.5, color='blue', alpha=0.7)
ax1.plot(df['Date'], df['MA_20'], label='20-day MA', linewidth=1.5, color='red', alpha=0.7)

# Plot buy/sell signals
buy_signals_df = df[df['Signal'] == 1]
sell_signals_df = df[df['Signal'] == -1]
ax1.scatter(buy_signals_df['Date'], buy_signals_df['Close'], 
           color='green', marker='^', s=150, label='Buy Signal', zorder=5)
ax1.scatter(sell_signals_df['Date'], sell_signals_df['Close'], 
           color='red', marker='v', s=150, label='Sell Signal', zorder=5)

ax1.set_title('MA Crossover Strategy - Trading Signals', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Chart 2: Portfolio Value Over Time
ax2 = axes[1]
ax2.plot(df['Date'], df['Portfolio_Value'], label='Strategy Portfolio', 
        linewidth=2, color='green', alpha=0.8)

# Calculate buy & hold portfolio value for comparison
buy_hold_portfolio = (df['Close'] / df['Close'].iloc[0]) * initial_capital
ax2.plot(df['Date'], buy_hold_portfolio, label='Buy & Hold', 
        linewidth=2, color='blue', alpha=0.6, linestyle='--')

ax2.axhline(y=initial_capital, color='black', linestyle=':', alpha=0.5, label='Initial Capital')
ax2.set_title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Chart 3: Drawdown
ax3 = axes[2]
ax3.fill_between(df['Date'], df['Drawdown'], 0, color='red', alpha=0.3)
ax3.plot(df['Date'], df['Drawdown'], color='darkred', linewidth=1.5)
ax3.set_title('Strategy Drawdown (Peak-to-Trough Decline)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Drawdown (%)', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/backtest_results.png', dpi=300, bbox_inches='tight')
plt.show()



# Save trade log to CSV
if len(trade_df) > 0:
    trade_df.to_csv('data/trade_log.csv', index=False)
    print(f"\n✅ Trade log saved to 'data/trade_log.csv'")
    print("\n📋 TRADE LOG (First 5 trades):")
    print(trade_df.head().to_string(index=False))

print("\n" + "=" * 80)
print("🎉 BACKTESTING COMPLETE!")
print("=" * 80)
print("\n💡 KEY INSIGHTS:")
print("   1. This is a REALISTIC backtest with transaction costs and slippage")
print("   2. We never used future data (no look-ahead bias)")
print("   3. We included risk management (stop-loss & take-profit)")
print("   4. We compared against buy-and-hold benchmark")
print("   5. All trades are logged and can be reviewed")
print("\n📊 Charts saved as 'data/backtest_results.png'")
print("=" * 80)