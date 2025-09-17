import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_hourly_data(ticker):
    stock = yf.Ticker(ticker)
    # Fetch past 5 days of hourly data (interval='1h')
    data = stock.history(period='5d', interval='30m')
    return data



st.title("Daily Penny Stock Charts (Price + Volume)")
stockspenny='TLRY,PGEN,TSLZ,PLUG,FLYE,LCID,DNN,CGC,IOBT,CUPR,BITF,IQ,AGL,VMAR,NVDQ'
#'MNTS,DNN,TLRY,CGTX,XFOR,PLUG,OPEN,TSLZ,VOR,INVZ,BITF,CRE,LCID,CGC,PACB'
# User inputs: comma-separated tickers
tickers_str = st.text_input(stockspenny,stockspenny)
tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
print(tickers)

#for ticker in tickers:
    #print(f"Fetching hourly data for {ticker}:")
    #print(df[['Close', 'Volume']].tail(24))  # Show last 24 hourly bars
# Set date range (defaults to recent 5 trading days)
start_date = st.date_input("Start date", pd.Timestamp.today() - pd.Timedelta(days=5))
end_date = st.date_input("End date", pd.Timestamp.today())
for ticker in tickers:
    st.subheader(f"Ticker: {ticker}")

    # Fetch daily OHLCV
    try:
        df = fetch_hourly_data(ticker)
        #df = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True)
        if df.empty:
            st.warning(f"No data for {ticker}.")
            continue
        #print(df)
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(df.index, df['Close'], color='blue', label='Close Price')
        ax1.set_ylabel('Price (USD)', color='blue')
        ax1.legend(loc='upper left')

        # Twin axis for volume
        ax2 = ax1.twinx()
        ax2.bar(df.index, df['Volume'], color='gray', alpha=0.3, label='Volume')
        ax2.set_ylabel('Volume', color='gray')
        ax2.legend(loc='upper right')
        
        plt.title(f"{ticker} Daily Price and Volume")
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(df[['Close','Volume']])
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")

