# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from swingtradeML import SwingTradingML
from datetime import datetime

st.set_page_config(page_title="Swing Trading ML", layout="wide", page_icon="📈")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: bold; color: #1f77b4;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 Swing Trading ML System</p>', unsafe_allow_html=True)
st.markdown("**Machine Learning-powered swing trading with ensemble methods**")

# Sidebar
st.sidebar.header("⚙️ Configuration")
symbol = st.sidebar.text_input("Ticker Symbol", value="SPY")
lookback_days = st.sidebar.slider("Lookback Days", 365, 1825, 730)
forward_days = st.sidebar.slider("Forward Days (Swing Period)", 3, 10, 5)
threshold = st.sidebar.slider("Signal Threshold (%)", 0.5, 5.0, 2.0) / 100

train_button = st.sidebar.button("🚀 Train Model", type="primary")

# Main content
col1, col2, col3, col4 = st.columns(4)

if train_button:
    with st.spinner("Training model... This may take a minute."):
        try:
            # Initialize trader
            trader = SwingTradingML(symbol=symbol, lookback_days=lookback_days)
            
            # Download and process data
            df = trader.download_data()
            df = trader.engineer_features(df)
            df = trader.create_labels(df, forward_days=forward_days, threshold=threshold)
            X, y, data = trader.prepare_ml_data(df)
            
            # Store in session state
            st.session_state['trader'] = trader
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['data'] = data
            
            # Train
            with st.expander("📊 Training Log", expanded=False):
                model = trader.train_model(X, y)
            
            # Predictions
            predictions, proba = trader.predict(X)
            results = trader.backtest(data, predictions)
            
            st.session_state['results'] = results
            st.session_state['predictions'] = predictions
            st.session_state['proba'] = proba
            
            st.success("✅ Model trained successfully!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Display results if available
if 'results' in st.session_state:
    results = st.session_state['results']
    trader = st.session_state['trader']
    predictions = st.session_state['predictions']
    
    # Metrics
    total_return_market = (results['cumulative_market_returns'].iloc[-1] - 1) * 100
    total_return_strategy = (results['cumulative_strategy_returns'].iloc[-1] - 1) * 100
    sharpe_strategy = results['strategy_returns'].mean() / results['strategy_returns'].std() * np.sqrt(252)
    max_drawdown_strategy = (results['cumulative_strategy_returns'] / results['cumulative_strategy_returns'].cummax() - 1).min() * 100
    
    col1.metric("Market Return", f"{total_return_market:.2f}%")
    col2.metric("Strategy Return", f"{total_return_strategy:.2f}%", 
                delta=f"{total_return_strategy - total_return_market:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe_strategy:.2f}")
    col4.metric("Max Drawdown", f"{max_drawdown_strategy:.2f}%")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🎯 Signals", "🔍 Feature Importance", "📊 Latest Prediction"])
    
    with tab1:
        # Cumulative returns chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results.index, y=results['cumulative_market_returns'],
                                 name='Buy & Hold', line=dict(color='gray', width=2)))
        fig.add_trace(go.Scatter(x=results.index, y=results['cumulative_strategy_returns'],
                                 name='ML Strategy', line=dict(color='#1f77b4', width=2)))
        fig.update_layout(title='Cumulative Returns Comparison',
                         xaxis_title='Date', yaxis_title='Cumulative Returns',
                         height=500, hovermode='x unified')
        st.plotly_chart(fig, width='stretch')
        
        # Drawdown
        drawdown = (results['cumulative_strategy_returns'] / results['cumulative_strategy_returns'].cummax() - 1) * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=results.index, y=drawdown, fill='tozeroy',
                                  name='Drawdown', line=dict(color='red')))
        fig2.update_layout(title='Strategy Drawdown', xaxis_title='Date', 
                          yaxis_title='Drawdown (%)', height=300)
        st.plotly_chart(fig2, width='stretch')
    
    with tab2:
        # Price with signals
        buy_signals = results[results['predicted_signal'] == 1]
        sell_signals = results[results['predicted_signal'] == -1]
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=results.index, y=results['Close'],
                                  name='Close Price', line=dict(color='black')))
        fig3.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                  mode='markers', name='Buy Signal',
                                  marker=dict(color='green', size=10, symbol='triangle-up')))
        fig3.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                  mode='markers', name='Sell Signal',
                                  marker=dict(color='red', size=10, symbol='triangle-down')))
        fig3.update_layout(title=f'{symbol} Price with Trading Signals',
                          xaxis_title='Date', yaxis_title='Price', height=500)
        st.plotly_chart(fig3, width='stretch')
        
        # Signal distribution
        signal_counts = pd.Series(predictions).value_counts().sort_index()
        signal_labels = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Buy Signals", int(signal_counts.get(1, 0)))
            st.metric("Total Sell Signals", int(signal_counts.get(-1, 0)))
        with col_b:
            st.metric("Total Hold Signals", int(signal_counts.get(0, 0)))
            win_rate = (results['strategy_returns'] > 0).sum() / len(results) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with tab3:
        # Feature importance
        importance_df = trader.get_feature_importance(top_n=20)
        
        fig4 = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker=dict(color='#1f77b4')
        ))
        fig4.update_layout(title='Top 20 Feature Importances',
                          xaxis_title='Importance', yaxis_title='Feature',
                          height=600, yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig4, width='stretch')
    
    with tab4:
        # Latest prediction
        latest_data = results.iloc[-1]
        latest_pred = predictions[-1]
        latest_proba = st.session_state['proba'][-1]
        
        signal_map = {-1: "🔴 SELL", 0: "⚪ HOLD", 1: "🟢 BUY"}
        color_map = {-1: "red", 0: "gray", 1: "green"}
        
        st.markdown(f"### Latest Signal for {symbol}")
        st.markdown(f"<h1 style='color:{color_map[latest_pred]}'>{signal_map[latest_pred]}</h1>", 
                   unsafe_allow_html=True)
        
        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Current Price", f"${latest_data['Close']:.2f}")
        col_y.metric("Date", latest_data.name.strftime('%Y-%m-%d'))
        
        # Probabilities
        st.markdown("#### Signal Probabilities")
        prob_df = pd.DataFrame({
            'Signal': ['Sell', 'Hold', 'Buy'],
            'Probability': [f"{latest_proba[0]:.1%}", f"{latest_proba[1]:.1%}", f"{latest_proba[2]:.1%}"]
        })
        st.dataframe(prob_df, width='stretch', hide_index=True)
        
        # Recent indicators
        st.markdown("#### Key Technical Indicators")
        indicator_data = {
            'Indicator': ['RSI (14)', 'MACD', 'BB Position', 'ADX', 'Volume Ratio'],
            'Value': [
                f"{latest_data['RSI_14']:.2f}",
                f"{latest_data['MACD']:.4f}",
                f"{latest_data['BB_position']:.2f}",
                f"{latest_data['ADX']:.2f}",
                f"{latest_data['volume_ratio']:.2f}"
            ]
        }
        st.dataframe(pd.DataFrame(indicator_data), width='stretch', hide_index=True)

else:
    st.info("👈 Configure parameters in the sidebar and click 'Train Model' to begin")
    
    st.markdown("""
    ### Features
    - **Advanced ML**: LightGBM with time-series cross-validation
    - **30+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, Volume, etc.
    - **Ensemble Approach**: Captures multiple market conditions
    - **Real-time Predictions**: Latest trading signals with probabilities
    - **Performance Analytics**: Sharpe ratio, drawdown, win rate
    
    ### How It Works
    1. Downloads historical price data for the selected ticker
    2. Engineers 30+ technical features from price/volume
    3. Creates swing trading labels (buy/sell/hold) based on forward returns
    4. Trains LightGBM model using time-series cross-validation
    5. Generates trading signals and backtests performance
    """)
