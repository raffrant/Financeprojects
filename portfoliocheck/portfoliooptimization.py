import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, expected_returns, covariance_matrix, risk_aversion=1.0):
        """
        Initialize the optimizer with expected returns, covariance of returns,
        and a parameter to balance risk versus return.

        Parameters:
        - expected_returns: np.array of annualized returns for each stock
        - covariance_matrix: np.array covariance matrix of annualized returns
        - risk_aversion: float, higher means more cautious about risk
        """
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_aversion = risk_aversion
        self.num_assets = len(expected_returns)
        self.weights = None

    def optimize(self):
        """
        Find the portfolio weights that maximize return for a given risk appetite.
        This solves the classic mean-variance optimization problem.
        """
        def objective(weights):
            # Negative because we minimize in scipy
            expected_portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
            return -expected_portfolio_return + self.risk_aversion * portfolio_variance

        # The weights have to sum to 1 (all your money allocated)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

        # No short selling allowed; weights between 0 and 1
        bounds = [(0, 1) for _ in range(self.num_assets)]

        # Start with equal weights as a guess
        initial_weights = np.ones(self.num_assets) / self.num_assets

        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise Exception("Optimization failed!")

    def print_weights(self, stock_names):
        """
        Nicely print out each stock's weight as a percentage.
        """
        print(f"\nPortfolio allocation with risk aversion = {self.risk_aversion}:")
        for name, weight in zip(stock_names, self.weights):
            print(f"  {name}: {weight*100:.2f}%")

# Fetch stock price data from Yahoo Finance
stocks = ["AIQ","BOTZ","ROBO","ARKK","SMH","WTAI","XBI","IBB","FBT","SBIO","BBH","XLF","KBE","FINX","IPAY"]#['AAPL', 'MSFT', 'GOOG'] this is where you can play around with your stocks
start = '2010-01-01'
end = '2025-01-01'

print("Downloading historical data for stocks...")
price_data = yf.download(stocks, start=start, end=end)['Close']

# Calculate daily returns and then annualized expected returns and covariance matrix
daily_returns = price_data.pct_change().dropna()
annual_return = daily_returns.mean() * 252
annual_covariance = daily_returns.cov() * 252

# Create optimizer instance with a chosen risk aversion level
risk_aversion_level = 1.0  # Adjust this to be more or less conservative
optimizer = PortfolioOptimizer(annual_return.values, annual_covariance.values, risk_aversion_level)

print("Running portfolio optimization...")
best_weights = optimizer.optimize()
optimizer.print_weights(stocks)

# Let's compare some portfolio strategies
strategies = [
    ('Mean-Variance Optimal', best_weights),
    ('Maximum Return Only', np.eye(len(stocks))[np.argmax(annual_return.values)]),
    ('Equal Weight', np.ones(len(stocks)) / len(stocks))
]

# Add single-stock portfolios for comparison
for stock in stocks:
    weights = [1 if s == stock else 0 for s in stocks]
    strategies.append((f"All {stock}", weights))

# Calculate expected return and risk for each strategy
records = []
for name, weights in strategies:
    ret = np.dot(weights, annual_return.values)
    risk = np.dot(weights, np.dot(annual_covariance.values, weights))
    records.append({'Strategy': name, 'Return': ret, 'Risk (Variance)': risk})

df_results = pd.DataFrame(records)

# Plotting the results
fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()

df_results.plot(x='Strategy', y='Return', kind='bar', ax=ax1, color='dodgerblue', legend=False)
df_results.plot(x='Strategy', y='Risk (Variance)', kind='line', marker='o', ax=ax2, color='darkorange', legend=False)

ax1.set_xlabel('')
ax1.set_ylabel('Annualized Return', color='dodgerblue')
ax2.set_ylabel('Annualized Risk (Variance)', color='darkorange')
ax1.set_title('Portfolio Strategies: Return vs Risk')

ax1.tick_params(axis='y', labelcolor='dodgerblue')
ax2.tick_params(axis='y', labelcolor='darkorange')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
