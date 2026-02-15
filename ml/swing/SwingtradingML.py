# swing_trading_ml.py
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import ta
import warnings
warnings.filterwarnings('ignore')

class SwingTradingML:
    def __init__(self, symbol='SPY', lookback_days=1240):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.model = None
        self.feature_columns = []
        
    def download_data(self):
        """Download historical data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        df = yf.download(self.symbol, start=start_date, end=end_date, progress=False)
        
        # FIX: Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # FIX: Ensure columns are standard and remove any extra dimensions
        df = df.copy()
        for col in df.columns:
            if df[col].ndim > 1:
                df[col] = df[col].iloc[:, 0]
        
        return df
    
    def engineer_features(self, df):
        """Create technical indicators and features"""
        data = df.copy()
        
        # FIX: Ensure all columns are 1D Series
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = data[col].squeeze()  # Convert to 1D if needed
        
        # Extract as 1D Series explicitly
        close = data['Close'].squeeze()
        high = data['High'].squeeze()
        low = data['Low'].squeeze()
        volume = data['Volume'].squeeze()
        
        # Price-based features
        data['returns'] = close.pct_change()
        data['log_returns'] = np.log(close / close.shift(1))
        
        # Moving Averages
        data['SMA_10'] = ta.trend.sma_indicator(close, window=10)
        data['SMA_20'] = ta.trend.sma_indicator(close, window=20)
        data['SMA_50'] = ta.trend.sma_indicator(close, window=50)
        data['EMA_12'] = ta.trend.ema_indicator(close, window=12)
        data['EMA_26'] = ta.trend.ema_indicator(close, window=26)
        
        # Price vs MA ratios
        data['price_to_sma20'] = close / data['SMA_20']
        data['price_to_sma50'] = close / data['SMA_50']
        
        # MACD
        macd = ta.trend.MACD(close)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_diff'] = macd.macd_diff()
        
        # RSI
        data['RSI_14'] = ta.momentum.rsi(close, window=14)
        data['RSI_7'] = ta.momentum.rsi(close, window=7)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close, window=20)
        data['BB_high'] = bollinger.bollinger_hband()
        data['BB_low'] = bollinger.bollinger_lband()
        data['BB_mid'] = bollinger.bollinger_mavg()
        data['BB_width'] = (data['BB_high'] - data['BB_low']) / data['BB_mid']
        data['BB_position'] = (close - data['BB_low']) / (data['BB_high'] - data['BB_low'])
        
        # ATR (Volatility)
        data['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
        
        # Volume features
        volume_ma = volume.rolling(window=20).mean()
        data['volume_ratio'] = volume / volume_ma
        data['OBV'] = ta.volume.on_balance_volume(close, volume)
        data['OBV_change'] = data['OBV'].pct_change()
        
        # ADX (Trend Strength)
        data['ADX'] = ta.trend.adx(high, low, close, window=14)
        
        # Momentum
        data['momentum_5'] = close / close.shift(5) - 1
        data['momentum_10'] = close / close.shift(10) - 1
        
        # Volatility
        data['volatility_20'] = data['returns'].rolling(window=20).std()
        
        # Rate of Change
        data['ROC_10'] = ta.momentum.roc(close, window=10)
        
        return data
    
    def create_labels(self, df, forward_days=5, threshold=0.02):
        """Create swing trading labels"""
        data = df.copy()
        
        # Ensure Close is 1D
        close = data['Close'].squeeze()
        
        data['future_return'] = close.shift(-forward_days) / close - 1
        
        data['signal'] = 0
        data.loc[data['future_return'] > threshold, 'signal'] = 1
        data.loc[data['future_return'] < -threshold, 'signal'] = -1
        
        return data
    
    def prepare_ml_data(self, df):
        """Prepare features and labels for ML"""
        # Drop rows with NaN
        data = df.dropna().copy()
        
        # Define feature columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
                       'future_return', 'signal', 'OBV']
        self.feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Keep as DataFrame for features
        X = data[self.feature_columns].copy()
        
        # Extract target as 1D numpy array
        y = data['signal'].values.ravel()
        
        return X, y, data
    
    def train_model(self, X, y):
        """Train LightGBM model with time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        
        best_model = None
        best_score = 0
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 6
        }
        
        print(f"\n{'='*50}")
        print("Training LightGBM Model with Time Series Cross-Validation")
        print(f"{'='*50}\n")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Ensure y is 1D
            y_train = np.asarray(y_train).ravel()
            y_val = np.asarray(y_val).ravel()
            
            # Convert labels: -1, 0, 1 -> 0, 1, 2
            y_train_encoded = y_train + 1
            y_val_encoded = y_val + 1
            
            train_data = lgb.Dataset(X_train, label=y_train_encoded)
            val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
            )
            
            y_pred_encoded = model.predict(X_val, num_iteration=model.best_iteration)
            y_pred = np.argmax(y_pred_encoded, axis=1) - 1
            
            acc = accuracy_score(y_val, y_pred)
            print(f"Fold {fold + 1} - Validation Accuracy: {acc:.4f}")
            
            if acc > best_score:
                best_score = acc
                best_model = model
        
        print(f"\nBest Validation Accuracy: {best_score:.4f}")
        self.model = best_model
        return best_model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1) - 1
        
        return y_pred, y_pred_proba
    
    def backtest(self, data, predictions):
        """Simple backtest of the strategy"""
        results = data.copy()
        
        # Ensure predictions is 1D
        predictions = np.asarray(predictions).ravel()
        
        # Add predictions to results
        results['predicted_signal'] = predictions
        
        # Calculate returns
        results['strategy_returns'] = results['predicted_signal'].shift(1) * results['returns']
        results['cumulative_market_returns'] = (1 + results['returns']).cumprod()
        results['cumulative_strategy_returns'] = (1 + results['strategy_returns']).cumprod()
        
        return results
    
    def get_feature_importance(self, top_n=15):
        """Get top feature importances"""
        if self.model is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df

# Main execution
if __name__ == "__main__":
    # Initialize and train
    trader = SwingTradingML(symbol='SPY', lookback_days=730)
    
    print("Downloading data...")
    df = trader.download_data()
    print(f"Downloaded data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\nEngineering features...")
    df = trader.engineer_features(df)
    
    print("Creating labels...")
    df = trader.create_labels(df, forward_days=5, threshold=0.02)
    
    print("Preparing ML data...")
    X, y, data = trader.prepare_ml_data(df)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Number of features: {len(X.columns)}")
    print(f"Signal distribution:\n{pd.Series(y).value_counts().sort_index()}")
    
    # Train model
    model = trader.train_model(X, y)
    
    # Make predictions
    predictions, proba = trader.predict(X)
    
    # Backtest
    results = trader.backtest(data, predictions)
    
    # Calculate metrics
    total_return_market = (results['cumulative_market_returns'].iloc[-1] - 1) * 100
    total_return_strategy = (results['cumulative_strategy_returns'].iloc[-1] - 1) * 100
    
    # Handle NaN in strategy returns
    strategy_returns_clean = results['strategy_returns'].dropna()
    if len(strategy_returns_clean) > 0 and strategy_returns_clean.std() > 0:
        sharpe_strategy = strategy_returns_clean.mean() / strategy_returns_clean.std() * np.sqrt(252)
    else:
        sharpe_strategy = 0.0
    
    print(f"\n{'='*50}")
    print("BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"Market Return: {total_return_market:.2f}%")
    print(f"Strategy Return: {total_return_strategy:.2f}%")
    print(f"Strategy Sharpe Ratio: {sharpe_strategy:.2f}")
    
    # Feature importance
    print(f"\n{'='*50}")
    print("TOP FEATURES")
    print(f"{'='*50}")
    print(trader.get_feature_importance())
