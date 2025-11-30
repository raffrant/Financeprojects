import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simple_stop_optimizer(symbols, stop_levels=[2, 3, 4]):
    """Buy at open, test 2%/3%/4% stops. That's it."""
    results = []
    
    for symbol in symbols:
        print(f"\n{symbol}")
        data = yf.download(symbol, period="2y", progress=False)
        opens = data['Open'].values
        lows = data['Low'].values
        closes = data['Close'].values
        
        for stop_pct in stop_levels:
            # For each day: buy open → stop loss or hold to close
            pnl = []
            for o, l, c in zip(opens, lows, closes):
                if l <= o * (1 - stop_pct/100):
                    pnl.append(-stop_pct/100)  # Hit stop
                else:
                    y=(c - o) / o
                    pnl.append(y[0])    # Hold to close
            #pnl=[pnl[i][0] for i in range(len(pnl))]
            print(pnl)
            pnl = np.array(pnl)
            sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(252)
            
            results.append({
                'symbol': symbol,
                'stop_pct': f"{stop_pct}%",
                'sharpe': sharpe,
                'win_rate': np.mean(pnl > 0),
                'stop_hits': np.mean(pnl == -stop_pct/100)
            })
            print(f"  {stop_pct}% stop: Sharpe={sharpe:.2f}")
    
    return pd.DataFrame(results)

# RUN
symbols = ["AAPL", "MSFT", "NVDA"]
results = simple_stop_optimizer(symbols)

print("\n🏆 RESULTS:")
print(results.pivot(index='symbol', columns='stop_pct', values='sharpe').round(2))

# PLOT
pivot = results.pivot(index='symbol', columns='stop_pct', values='sharpe')
pivot.plot(kind='bar', title="Best Stop Loss %")
plt.ylabel('Sharpe Ratio')
plt.xticks(rotation=0)
plt.show()

# BEST
best = results.loc[results['sharpe'].idxmax()]
print(f"\n🎯 WINNER: {best['symbol']} {best['stop_pct']} → Sharpe {best['sharpe']:.2f}")
