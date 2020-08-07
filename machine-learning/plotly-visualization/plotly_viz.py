
# coding: utf-8

# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr

py.init_notebook_mode()


# In[ ]:


x = [ i for i in range(-10,10) ]

y = [ i*2 for i in range(-10,10) ]

xaxis = go.layout.XAxis(title="X Axis")
yaxis = go.layout.YAxis(title="Y Axis")

fig = go.Figure(layout=go.Layout(title="Simple Line Plot", xaxis=xaxis, yaxis=yaxis))
fig.add_trace(go.Scatter(x=x, y=y))


# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp((-1) * x))

x = sorted(np.random.random(100) * 10 - 5)
y = [ sigmoid(i) for i in x ]

xaxis = go.layout.XAxis(title="X Axis")
yaxis = go.layout.YAxis(title="Y Axis")

fig=go.Figure(layout=go.Layout(title="Sigmoid Plot",xaxis=xaxis, yaxis=yaxis))
fig.add_trace(go.Scatter(x=x, y=y, marker=dict(color="red")))


# In[ ]:


l = []

for _ in range(5):
    l.append([ sorted(np.random.randint(low=0, high=10000, size=50)), sorted(np.random.randint(low=0, high=10000, size=50)) ])

l = np.array(l)

figure = go.Figure(layout=go.Layout(title="Simple Scatter Example", xaxis=go.layout.XAxis(title="X"), yaxis=go.layout.YAxis(title="Y")))
for i in range(len(l)):
    figure.add_trace(go.Scatter(x=l[i][0],y=l[i][1], mode="markers", name=f" Distribution {i+1} "))
figure.show()


# In[ ]:


dist = np.random.normal(loc=0, scale=1, size=50000)


# In[ ]:


figure = go.Figure()
figure.add_trace(go.Histogram(x=dist,))


# In[ ]:




d=[{"values":np.random.normal(0,0.5,10000), "information": " Normal Distribution with mean 0 and std= 0.5"},
  {"values":np.random.normal(0,1,10000), "information": " Normal Distribution with mean 0 and std= 1"},
  {"values":np.random.normal(0,1.5,10000), "information": " Normal Distribution with mean 0 and std= 1.5"},
  {"values":np.random.normal(0,2,10000), "information": " Normal Distribution with mean 0 and std= 2"},
  {"values":np.random.normal(0,5,10000), "information": " Normal Distribution with mean 0 and std= 5"}]

ff.create_distplot([ele["values"] for ele in d], group_labels=[ele["information"] for ele in d], show_hist=False)


# In[ ]:


x = np.random.randint(low=5, high=100, size=15)
y = np.random.randint(low=5, high=100 ,size=15)
z = np.random.randint(low=5, high=100, size=15)

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers"))


# In[ ]:


df_iris = pd.read_csv("iris.csv")


# In[ ]:


fig = go.Figure()
species_types = df_iris.species.unique().tolist()

for specie in species_types:
    b = df_iris.species == specie
    fig.add_trace(go.Scatter3d(x=df_iris["sepal_length"][b], y=df_iris["sepal_width"][b], z=df_iris["petal_width"][b], name=specie, mode="markers"))


fig.show()


# In[ ]:


yf.pdr_override()

symbols = ["AAPL","MSFT"]
stocks = []
for symbol in symbols:
    stocks.append(pdr.get_data_yahoo(symbol, start="2020-01-01", end="2020-05-31"))


# In[ ]:


fig = go.Figure()

for stock,symbol in zip(stocks,symbols):
    fig.add_trace(go.Scatter(x=stock.index, y=stock.Close, name=symbol))

fig.show()


# In[ ]:


df_aapl = pdr.get_data_yahoo(symbol, start="2020-01-01", end="2020-05-31")


# In[ ]:


ff.create_candlestick(dates=df_aapl.index, open=df_aapl.Open, high=df_aapl.High, low=df_aapl.Low, close=df_aapl.Close)


# In[ ]:




