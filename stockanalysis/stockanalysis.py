#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:11:54 2018

@author: nick
"""

import datetime as dt
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
start=dt.datetime(2018,1,1)
end=dt.datetime(2018,4,1)
df=web.DataReader('TSLA', 'morningstar', start,end)
df.reset_index(inplace=True) #inplace sama kyk df=df.reset_index
df.set_index("Date", inplace=True)
df.to_csv('tsla.csv')
df=pd.read_csv('tsla.csv', parse_dates=True, index_col=0)
df['100ma']=df['Close'].rolling(window=100).mean()
print(df[['Open','High']].head())
print(df.tail())
df.plot()
plt.show()



