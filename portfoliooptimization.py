import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

class MeanVarianceOptimizer:
    def __init__(self, expected_returns, cov_matrix, risk_aversion=1.0):
        """
        expected_returns: numpy array of expected returns (annualized)
        cov_matrix: covariance matrix of the returns (annualized)
        risk_aversion: lambda parameter controlling tradeoff between risk and return
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_aversion = risk_aversion
        self.num_assets = len(expected_returns)
        self.weights = None

    def optimize(self):
        def objective(weights):
            # Mean-variance objective to minimize: -return + lambda * variance
            ret = np.dot(weights, self.expected_returns)
            var = np.dot(weights, np.dot(self.cov_matrix, weights))
            return -ret + self.risk_aversion * var

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * self.num_assets
        initial_guess = np.ones(self.num_assets) / self.num_assets

        result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise ValueError("Optimization failed")

    def summary(self, stock_names):
        print(f"Optimal portfolio weights (risk_aversion={self.risk_aversion}):")
        for name, w in zip(stock_names, self.weights):
            print(f"{name}: {w*100:.2f}%")

# 1. Download historical price data from Yahoo Finance
stock_names =["AIQ","BOTZ","ROBO","ARKK","SMH","WTAI","XBI","IBB","FBT","SBIO","BBH","XLF","KBE","FINX","IPAY"]# ['AAPL', 'MSFT', 'GOOG'] this line is for editing, put whatever stocks you like to see an optimized portfolio 
start_date = '2010-01-01'                    
end_date = '2025-01-01'
prices = yf.download(stock_names, start=start_date, end=end_date)['Close']

# 2. Calculate daily returns and annualize
returns = prices.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# 3. Optimize portfolio with risk-aversion parameter
risk_aversion = 1  # Adjust between 0 (max return only) and higher for more risk aversion
optimizer = MeanVarianceOptimizer(mean_returns.values, cov_matrix.values, risk_aversion)
optimal_weights = optimizer.optimize()
optimizer.summary(stock_names)

# 4. Prepare comparison data for visualization
cases = [
    ('Mean-Variance Optimal', optimal_weights),
    ('Max Return', np.eye(len(stock_names))[np.argmax(mean_returns)]),
    ('Equal Weight', np.ones(len(stock_names))/len(stock_names)),
]
for name in stock_names:
    weights = [1 if s == name else 0 for s in stock_names]
    cases.append((f'All {name}', weights))

viz_data = {'Strategy': [], 'Portfolio Return': [], 'Portfolio Variance': []}
for label, w in cases:
    port_return = np.dot(w, mean_returns.values)
    port_var = np.dot(w, np.dot(cov_matrix.values, w))
    viz_data['Strategy'].append(label)
    viz_data['Portfolio Return'].append(port_return)
    viz_data['Portfolio Variance'].append(port_var)

viz_df = pd.DataFrame(viz_data)

# 5. Plot returns and variances per strategy
fig, ax1 = plt.subplots(figsize=(9, 5))

ax2 = ax1.twinx()
bars = ax1.bar(viz_df['Strategy'], viz_df['Portfolio Return'], color='r', label='Return')
line = ax2.plot(viz_df['Strategy'], viz_df['Portfolio Variance'], color='g', marker='o', label='Variance')

ax1.set_ylabel('Annualized Portfolio Return', color='r')
ax2.set_ylabel('Annualized Portfolio Variance', color='g')
ax1.set_title('Portfolio Strategies: Return and Variance')

ax1.tick_params(axis='y', labelcolor='r')
ax2.tick_params(axis='y', labelcolor='g')
fig.autofmt_xdate(rotation=30)

# Create combined legend
bars_label = bars[0].get_label()
line_label = line[0].get_label()
handles = [bars, line[0]]
labels = ['Return', 'Variance']
ax1.legend(handles, labels, loc='upper right')

plt.tight_layout()
plt.show()
