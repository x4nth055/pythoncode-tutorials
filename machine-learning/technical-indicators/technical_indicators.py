# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import yfinance as yf
import pandas_datareader as pdr
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt


# %%
# import SPY stock price
df_spy = pdr.get_data_yahoo("SPY", start="2019-01-01", end="2019-09-30")
# import AAPL stock price
df_aapl = pdr.get_data_yahoo("AAPL", start="2019-01-01", end="2019-09-30")


# %%
df_spy.head()


# %%
df_aapl[["Open", "High", "Low", "Close"]].plot()
plt.show()


# %%
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot()

plot_data = []
for i in range(150, len(df_aapl)):
    row = [
        i, 
        df_aapl.Open.iloc[i], 
        df_aapl.High.iloc[i], 
        df_aapl.Low.iloc[i], 
        df_aapl.Close.iloc[i], 
    ]
    plot_data.append(row)
candlestick_ohlc(ax, plot_data)
plt.show()


# %%
from stockstats import StockDataFrame
stocks = StockDataFrame.retype(df_aapl[["Open", "Close", "High", "Low", "Volume"]])


# %%
plt.plot(stocks["close_10_sma"], color="b", label="SMA")
plt.plot(df_aapl.Close, color="g", label="Close prices")
plt.legend(loc="lower right")
plt.show()


# %%
plt.plot(stocks["close_10_sma"], color="b", label="SMA") # plotting SMA
plt.plot(stocks["close_10_ema"], color="k", label="EMA")
plt.plot(df_aapl.Close, color="g", label="Close prices") # plotting close prices
plt.legend(loc="lower right")
plt.show()


# %%
plt.plot(stocks["macd"], color="b", label="MACD")
plt.plot(stocks["macds"], color="g", label="Signal Line")
plt.legend(loc="lower right")
plt.show()


