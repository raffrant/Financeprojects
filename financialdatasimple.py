import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import yfinance as yf

#stocks=dict(
#    AAPL='Apple Stock',
#    AMZN='Amazon Stock',
#    CNY='CNY exchange rate',
#    GDX='Gold Price',
#    GLD='SPDR Gold Trust',
#    GS='Goldman Sachs Stock',
#    INTC='Intel Stock',
#    MSFT='Microsoft Stock',
#    SPY='S&P 500'
#)
class datafinance(object):
    def __init__(self,stocks):
        self.__stocks=stocks
    def data(self):
        df=yf.download(list(self.__stocks.keys()))
        df.dropna()
        x=df.info()
        adj_close=df['Adj Close']
        adj_close.dropna()
        y2=adj_close.head() # simple head to check the data if you want
        return df,adj_close
    
    def plot(self):
        yas,y2=self.data()
        yas.plot(subplots=True,figsize=(12,10))
        plt.show()
    
    def  descstats(self):   
        yfd1,yfd2=self.data()
        a1=yfd2.describe().round(2)
        a4=yfd2.pct_change().round(3).head()
        #yfd2.pct_change().mean().plot(kind='bar',figsize=(10,8)) #for plotting
        plt.show()
        return a1,a4

stc=dict(AAPL='Apple Stock',AMZN='Amazon Stock')
print(datafinance(stc).descstats())
