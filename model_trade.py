#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:35:32 2020

@author: alex
"""

import numpy as np
import pandas as pd



from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import time
from datetime import datetime, timedelta
import itertools
import os
import pickle
import math
from sklearn.preprocessing import StandardScaler
from ressup import ressup
import nest_asyncio
nest_asyncio.apply()


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

class get_data:

    def next_exp_weekday(self):
        weekdays = {2: [5, 6, 0], 4: [0, 1, 2], 0: [3, 4]}
        today = datetime.today().weekday()
        for exp, day in weekdays.items():
            if today in day:
                return exp

    def next_weekday(self, d, weekday):

        days_ahead = weekday - d.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        date_to_return = d + timedelta(days_ahead)  # 0 = Monday, 1=Tuself.ESday, 2=Wednself.ESday...
        return date_to_return.strftime('%Y%m%d')

    def get_strikes_and_expiration(self):
        ES = Future(symbol='ES', lastTradeDateOrContractMonth='20200918', exchange='GLOBEX',
                                currency='USD')
        ib.qualifyContracts(ES)
        expiration = self.next_weekday(datetime.today(), self.next_exp_weekday())
        chains = ib.reqSecDefOptParams(underlyingSymbol='ES', futFopExchange='GLOBEX', underlyingSecType='FUT',underlyingConId=ES.conId)
        chain = util.df(chains)
        strikes = chain[chain['expirations'].astype(str).str.contains(expiration)].loc[:, 'strikes'].values[0]
        [ESValue] = ib.reqTickers(ES)
        ES_price= ESValue.marketPrice()
        strikes = [strike for strike in strikes
                if strike % 5 == 0
                and ES_price - 10 < strike < ES_price + 10]
        return strikes,expiration

    def get_contract(self, right, net_liquidation):
        strikes, expiration=self.get_strikes_and_expiration()
        for strike in strikes:
            contract=FuturesOption(symbol='ES', lastTradeDateOrContractMonth=expiration,
                                                strike=strike,right=right,exchange='GLOBEX')
            ib.qualifyContracts(contract)
            price = ib.reqMktData(contract,"",False,False)
            if float(price.last)*50 >=net_liquidation:
                continue
            else:
                return contract

    def res_sup(self,ES_df):
        ES_df = ES_df.reset_index(drop=True)
        ressupDF = ressup(ES_df, len(ES_df))
        res = ressupDF['Resistance'].values
        sup = ressupDF['Support'].values
        return res, sup

    def ES(self):
        ES = Future(symbol='ES', lastTradeDateOrContractMonth='20200918', exchange='GLOBEX',
                                currency='USD')
        ib.qualifyContracts(ES)
        ES_df = ib.reqHistoricalData(contract=ES, endDateTime=endDateTime, durationStr=No_days,
                                     barSizeSetting=interval, whatToShow = 'TRADES', useRTH = False)
        ES_df = util.df(ES_df)
        ES_df.set_index('date',inplace=True)
        ES_df['Resistance'], ES_df['Support'] = self.res_sup(ES_df)
        ES_df['RSI'] = ta.RSI(ES_df['close'])
        ES_df['macd'],ES_df['macdsignal'],ES_df['macdhist'] = ta.MACD(ES_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        ES_df['MA_9']=ta.MA(ES_df['close'], timeperiod=9)
        ES_df['MA_21']=ta.MA(ES_df['close'], timeperiod=21)
        ES_df['MA_200']=ta.MA(ES_df['close'], timeperiod=200)
        ES_df['EMA_9']=ta.EMA(ES_df['close'], timeperiod=9)
        ES_df['EMA_21']=ta.EMA(ES_df['close'], timeperiod=21)
        ES_df['EMA_50']=ta.EMA(ES_df['close'], timeperiod=50)
        ES_df['EMA_200']=ta.EMA(ES_df['close'], timeperiod=200)
        ES_df['ATR']=ta.ATR(ES_df['high'],ES_df['low'], ES_df['close'])
        ES_df['roll_max_cp']=ES_df['high'].rolling(20).max()
        ES_df['roll_min_cp']=ES_df['low'].rolling(20).min()
        ES_df['roll_max_vol']=ES_df['volume'].rolling(20).max()
        ES_df['EMA_21-EMA_9']=ES_df['EMA_21']-ES_df['EMA_9']
        ES_df['EMA_200-EMA_50']=ES_df['EMA_200']-ES_df['EMA_50']
        ES_df['B_upper'], ES_df['B_middle'], ES_df['B_lower'] = ta.BBANDS(ES_df['close'], matype=MA_Type.T3)
        ES_df.dropna(inplace = True)
        
        return ES_df

    def option_history(self, contract):
        df = pd.DataFrame(util.df(ib.reqHistoricalData(contract=contract, endDateTime=endDateTime, durationStr=No_days,
                                      barSizeSetting=interval, whatToShow = 'MIDPOINT', useRTH = False, keepUpToDate=False))[['date','close']])
        df.columns=['date',f"{contract.symbol}_{contract.right}_close"]
        df.set_index('date',inplace=True)
        return df

    def options(self, df1,df2):
        return pd.merge(df1,df2, on='date', how='outer').dropna()

def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=5):
    """ A multi-layer perceptron """
     
    # input layer
    i = Input(shape=(input_dim,1))
    x = i
     
    # hidden layers
    for _ in range(n_hidden_layers):
      # x = Dropout(0.2)(x)
      # x = LSTM(hidden_dim, return_sequences = True)(x)
      x = Dense(hidden_dim, activation='relu')(x)
     
    x = GlobalAveragePooling1D()(x)
    # final layer
    # x = Dense(n_action, activation='relu')(x)
    x = Dense(n_action, activation='softmax')(x)
    # make the model
    model = Model(i, x)
     
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print((model.summary()))
    return model

models_folder = './RL_trade_ES_futures/rl_trader_models_Sup/1_layer_BO_RSI_ATR_Close'
name = f'{models_folder}/dqn.h5'
rewards_folder = './RL_trade_ES_futures/rl_trader_rewards_Sup/1_layer_BO_RSI_ATR_Close'
model = mlp(10,9)
model.load_weights(name)
previous_action = ''

with open(f'{rewards_folder}/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f) 

stock_owned = np.zeros(2)


def reset(data, stock_owned, cash_in_hand):
    stock_price = data.iloc[-1,-2:].values
    return _get_obs(stock_owned, stock_price, cash_in_hand)

def _get_obs(stock_owned, stock_price, cash_in_hand):
    obs = np.empty(10)
    obs[:2] = stock_owned
    obs[2:2*2] = stock_price
    obs[4] = cash_in_hand
    obs[5:] = data.iloc[-1,:-2]
    return obs, stock_price, cash_in_hand



from ib_insync import *
import talib as ta
from talib import MA_Type
ib = IB()
ib.disconnect()
ib.connect('127.0.0.1', 7497, clientId=np.random.randint(10, 1000))
ES = Future(symbol='ES', lastTradeDateOrContractMonth='20200918', exchange='GLOBEX',
            currency='USD')
ib.qualifyContracts(ES)
endDateTime = ''
No_days = '1 D'
interval = '1 min'
res = get_data()

    
    
def flatten_position(position):
    contract = position.contract
    if position.position > 0: # Number of active Long positions
        action = 'Sell' # to offset the long positions
    elif position.position < 0: # Number of active Short positions
        action = 'Buy' # to offset the short positions
    else:
        assert False
    totalQuantity = abs(position.position)
    order = MarketOrder(action=action, totalQuantity=totalQuantity)
    trade = ib.placeOrder(contract, order)
    print(f'Flatten Position: {action} {totalQuantity} {contract.localSymbol}')
    t0 = datetime.now()
    
    while trade.orderStatus.status != "Filled" :
        if (datetime.now() - t0).seconds > 120:
            ib.cancelOrder()
            return       
    assert trade in ib.trades(), 'trade not listed in ib.trades'
    
def option_position():
    stock_owned = np.zeros(2)
    position = ib.portfolio()
    call_contract= 0
    put_contract = 0
    for each in position:
        if str(res.get_contract('C', 2000).conId) in str(each.contract.conId):
            call_contract = each
            stock_owned[0] = each.position
        elif str(res.get_contract('P', 2000).conId) in str(each.contract.conId):
            put_contract = each
            stock_owned[1] = each.position
    return stock_owned, call_contract, put_contract
while True:    
    cash_in_hand = float(ib.accountSummary()[5].value)
    data_raw = res.options(res.options(res.ES(),res.option_history(res.get_contract('C', 2000)))\
                       ,res.option_history(res.get_contract('P', 2000))) 
    data = data_raw[['close', 'B_middle', 'B_lower', 'RSI', 'ATR', 'ES_C_close','ES_P_close']]
    stock_owned, call_contract, put_contract = option_position()
    state, stock_price, cash_in_hand = reset(data, stock_owned, cash_in_hand)
    state = scaler.transform(state.reshape(-1,10))
    action_list = list(map(list, itertools.product([0, 1, 2], repeat=2)))
    action=np.argmax(model.predict(state))
    # if previous_action == action:
    #     continue
    # previous_action = action
    action_vec = action_list[action]
    buy_index = [] 
    sell_index = []
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    if sell_index:
      for i in sell_index:
        if not stock_owned[i] == 0:
            position = call_contract if i == 0 else put_contract
            flatten_position(position)
        print('sell loop finished')
        cash_in_hand = float(ib.accountSummary()[5].value)
        stock_owned, call_contract, put_contract = option_position()
    if buy_index:
      can_buy = True
      while can_buy:
        for i in buy_index:
          if cash_in_hand > stock_price[i] * 50:
            contract = call_contract if i == 0 else put_contract
            order = LimitOrder('BUY', 1, stock_price[i] + 0.25)
            trade = ib.placeOrder(contract, order)
            t0 = datetime.now()
            while trade.orderStatus.status != "Filled" :
                if (datetime.now() - t0).seconds > 120:
                    ib.cancelOrder()
                    print('trade canceled')
                    break
            print('buy loop finished')
            stock_owned, call_contract, put_contract = option_position()
            cash_in_hand = float(ib.accountSummary()[5].value)
          else:
            can_buy = False
            
    print(action, action_vec, stock_owned, cash_in_hand)
    time.sleep(60)
