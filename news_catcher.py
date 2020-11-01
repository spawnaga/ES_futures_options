#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 14:25:50 2020

@author: alex
"""
import pandas as pd
from ib_insync import *
util.startLoop()
ib = IB()
if not ib.isConnected():
    ib.connect('127.0.0.1', 7497, 6)
newsProviders = ib.reqNewsProviders()
amd = Stock('SPY', 'SMART', 'USD')
ib.qualifyContracts(amd)
print(newsProviders)
ES = Future(symbol='ES', lastTradeDateOrContractMonth='20201218', exchange='GLOBEX',
                    currency='USD')
ib.qualifyContracts(ES)
codes = '+'.join(np.code for np in newsProviders)
headlines = ib.reqHistoricalNews(amd.conId, 'BZ', '', '', 50)
for i in range(1, len(headlines)-1):
    print(headlines[i].headline[3:7] if headlines[i].headline[3:7][0] != 'n' else 0)
g = util.df(headlines)
g['rating'] = g['headline'].apply(lambda st: float(st[st.find("{")+3:st.find("}")]) if st[st.find("{")+3:st.find("}")][0] == "1" or st[st.find("{")+3:st.find("}")][0] == '-' or st[st.find("{")+3:st.find("}")][0] == '0' else float(0)) 
ES = ib.reqHistoricalData(contract=ES, endDateTime='', durationStr='4 D',
                                       barSizeSetting='1 min', whatToShow='TRADES', useRTH=False, keepUpToDate=True,
                                       timeout=10, formatDate=2)
df = util.df(ES)
df['date'] = pd.to_datetime(df['date'], utc = True)
g['time'] = pd.to_datetime(g['time'], utc = True)
g['time'] = g['time'].dt.floor('T')
m = pd.merge(df, g, how='outer',  left_on='date', right_on='time', copy=True)
# df.apply(g['rating'] if g['time'] == df['date'] else 0)

m.fillna(0,inplace = True)
