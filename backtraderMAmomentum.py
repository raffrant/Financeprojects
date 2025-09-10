import backtrader as bt
from datetime import datetime
import pandas as pd
import yfinance as yf

# Combined Strategy Class
class BreakoutHighRiskRewardWithMA(bt.Strategy):
    params = dict(
        lookback=40,    # Bars to look back for breakout
        risk_reward=2,  # Risk-reward ratio
        pfast=5,       # Fast MA period
        pslow=50,       # Slow MA period
        size=300 ,        # Position size
        rsi_period=70,
        rsi_threshold=10
    )

    def __init__(self):
        # Breakout indicators
        self.range_high = bt.ind.Highest(self.data.high, period=self.p.lookback)
        self.range_low = bt.ind.Lowest(self.data.low, period=self.p.lookback)
        # Moving averages for trend filter
        self.sma_fast = bt.ind.SMA(self.data.close, period=self.p.pfast)
        self.sma_slow = bt.ind.SMA(self.data.close, period=self.p.pslow)
        self.crossover = bt.ind.CrossOver(self.sma_fast, self.sma_slow)
        self.order = None
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        self.avg_vol = bt.indicators.SimpleMovingAverage(self.data.volume, period=20)
    def next(self):
        if not self.position:
            # Entry: Breakout and uptrend confirmed
            if self.data.close[0] > self.range_high[-1] and self.sma_fast[0] > self.sma_slow[0]:
                entry_price = self.data.close[0]
                stop_loss = self.range_low[-1]
                risk = entry_price - stop_loss
                take_profit = entry_price + risk * self.p.risk_reward
                self.order = self.buy_bracket(
                    price=entry_price,
                    stopprice=stop_loss,
                    limitprice=take_profit,
                    size=self.p.size
                )
            if self.data.close[0] > self.range_high[-1] and self.rsi[0] > self.p.rsi_threshold:
                self.buy()          

            if (self.data.close[0] > self.range_high[-1] and self.rsi[0] > self.p.rsi_threshold and self.data.volume[0] > self.avg_vol[0]):
                 self.buy()    
        else:
            # Optional: exit if trend reverses
            if self.crossover < 0:
                self.close()
              
tickers=["MSFT"]
start_date = '2015-10-01'
end_date = datetime.today().strftime('%Y-%m-%d')
dataframes = {}

for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if not df.empty:
        dataframes[ticker] = df

# Initialize Backtrader engine
cerebro = bt.Cerebro()
cerebro.broker.setcash(100000) #put the money you want.

# Add data feeds
for ticker, df in dataframes.items():
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data, name=ticker)

# Add the strategy
cerebro.addstrategy(BreakoutHighRiskRewardWithMA)
# Set up the Backtrader engine (Cerebro)
# Add analyzers for key metrics
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')       #check classic metrics for trading 
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

# Run the backtest
results = cerebro.run()
strat = results[0]

# Collect and print key metrics
final_value = cerebro.broker.getvalue()
sharpe = strat.analyzers.sharpe.get_analysis()
drawdown = strat.analyzers.drawdown.get_analysis()
returns = strat.analyzers.returns.get_analysis()
trades = strat.analyzers.trades.get_analysis()

print(f"Final Portfolio Value: ${final_value:.2f}")
print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
print(f"Total Return: {returns.get('rtot', 0) * 100:.2f}%")
print(f"Total Trades: {trades.total.closed}")

# Plot the results
cerebro.plot()




