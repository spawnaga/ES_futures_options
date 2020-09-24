#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:35:32 2020

@author: alex
"""

import numpy as np
import pandas as pd

import sys
import asyncio
from datetime import datetime, timedelta
import math
from ressup import ressup
from ib_insync import *

# import nest_asyncio

# nest_asyncio.apply()

sys.setrecursionlimit(10 ** 9)


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
        ES = Future(symbol='ES', lastTradeDateOrContractMonth='20201218', exchange='GLOBEX',
                    currency='USD')
        ib.sleep(0)
        ib.qualifyContracts(ES)

        expiration = self.next_weekday(datetime.today(), self.next_exp_weekday())
        chains = ib.reqSecDefOptParams(underlyingSymbol='ES', futFopExchange='GLOBEX', underlyingSecType='FUT',
                                       underlyingConId=ES.conId)
        chain = util.df(chains)
        strikes = chain[chain['expirations'].astype(str).str.contains(expiration)].loc[:, 'strikes'].values[0]
        [ESValue] = ib.reqTickers(ES)
        ES_price = ESValue.marketPrice()
        strikes = [strike for strike in strikes
                   if strike % 5 == 0
                   and ES_price - 10 < strike < ES_price + 10]
        return strikes, expiration

    def get_contract(self, right, net_liquidation):
        strikes, expiration = self.get_strikes_and_expiration()
        for strike in strikes:
            contract = FuturesOption(symbol='ES', lastTradeDateOrContractMonth=expiration,
                                     strike=strike, right=right, exchange='GLOBEX')
            ib.qualifyContracts(contract)
            price = ib.reqMktData(contract, "", False, False)
            if float(price.last) * 50 >= net_liquidation:
                continue
            else:
                return contract

    def res_sup(self, ES_df):
        ES_df = ES_df.reset_index(drop=True)
        ressupDF = ressup(ES_df, len(ES_df))
        res = ressupDF['Resistance'].values
        sup = ressupDF['Support'].values
        return res, sup

    def ES(self, ES):

        ES_df = util.df(ES)
        ES_df.set_index('date', inplace=True)
        ES_df.index = pd.to_datetime(ES_df.index)
        ES_df['hours'] = ES_df.index.strftime('%H').astype(int)
        ES_df['minutes'] = ES_df.index.strftime('%M').astype(int)
        ES_df['hours + minutes'] = ES_df['hours'] * 100 + ES_df['minutes']
        ES_df['Day_of_week'] = ES_df.index.dayofweek
        ES_df['Resistance'], ES_df['Support'] = self.res_sup(ES_df)
        ES_df['RSI'] = ta.RSI(ES_df['close'])
        ES_df['macd'], ES_df['macdsignal'], ES_df['macdhist'] = ta.MACD(ES_df['close'], fastperiod=12, slowperiod=26,
                                                                        signalperiod=9)
        ES_df['macd - macdsignal'] = ES_df['macd'] - ES_df['macdsignal']
        ES_df['MA_9'] = ta.MA(ES_df['close'], timeperiod=9)
        ES_df['MA_21'] = ta.MA(ES_df['close'], timeperiod=21)
        ES_df['MA_200'] = ta.MA(ES_df['close'], timeperiod=200)
        ES_df['EMA_9'] = ta.EMA(ES_df['close'], timeperiod=9)
        ES_df['EMA_21'] = ta.EMA(ES_df['close'], timeperiod=21)
        ES_df['EMA_50'] = ta.EMA(ES_df['close'], timeperiod=50)
        ES_df['EMA_200'] = ta.EMA(ES_df['close'], timeperiod=200)
        ES_df['ATR'] = ta.ATR(ES_df['high'], ES_df['low'], ES_df['close'])
        ES_df['roll_max_cp'] = ES_df['high'].rolling(20).max()
        ES_df['roll_min_cp'] = ES_df['low'].rolling(20).min()
        ES_df['roll_max_vol'] = ES_df['volume'].rolling(20).max()
        ES_df['vol/max_vol'] = ES_df['volume'] / ES_df['roll_max_vol']
        ES_df['EMA_21-EMA_9'] = ES_df['EMA_21'] - ES_df['EMA_9']
        ES_df['EMA_200-EMA_50'] = ES_df['EMA_200'] - ES_df['EMA_50']
        ES_df['B_upper'], ES_df['B_middle'], ES_df['B_lower'] = ta.BBANDS(ES_df['close'], matype=MA_Type.T3)
        ES_df.dropna(inplace=True)

        return ES_df


class all_events:
    def __init__(self):
        self.call_position = None
        self.put_position = None
        self.stock_owned = None
        self.canceltrade = None
        self.opentrade = None
        self.position = None
        self.orderstatus = None
        self.ordermodify = None

    def account_update(self, value=None):
        self.account = account

    def open_orders(self, trade=None):
        self.opentrade = trade
        return

    def connected(self):
        print('is connected')

    def disconnected(self):
        print('is disconnected')
        connect()

    def canceled_order(self, trade=None):
        self.canceltrade = trade

    def portfolio_update(self, PortfolioItem=None):
        self.portfolio = PortfolioItem
        # stock_owned = np.zeros(2)
        # position = [events.portfolio]
        # call_position = None
        # put_position = None
        # try:
        #     for each in position:
        #         if each.contract.right == 'C':
        #             call_position = each.contract
        #             ib.qualifyContracts(call_position)
        #             self.stock_owned[0] = each.position
        #         elif each.contract.right == 'P':
        #             put_position = each.contract
        #             ib.qualifyContracts(put_position)
        #             self.stock_owned[1] = each.position
        # except:
        #     self.call_position = self.stock_owned[0] if call_position != None else res.get_contract('C', 2000)
        #     self.put_position = self.stock_owned[1] if put_position != None else res.get_contract('P', 2000)

    def position_update(self, position=None):
        self.position = position

    def order_status(self, trade=None):
        self.orderstatus = trade

    def order_modify(self, trade=None):
        self.ordermodify = trade


def flatten_position(contract, price):
    print('flatttttttttttttttttttttttttttttttttttttttttttttttttttttt')
    portfolio = ib.portfolio()
    for each in portfolio:
        if each.contract.right != contract.right:
            continue
        ib.qualifyContracts(each.contract)
        if each.position > 0:  # Number of active Long portfolio
            action = 'SELL'  # to offset the long portfolio
        elif each.position < 0:  # Number of active Short portfolio
            action = 'BUY'  # to offset the short portfolio
        else:
            assert False
        totalQuantity = abs(each.position)

        print(f'price = {price.bid}')
        print(f'Flatten Position: {action} {totalQuantity} {contract.localSymbol}')
        order = LimitOrder(action, totalQuantity, price.bid - 0.25)
        trade = ib.placeOrder(each.contract, order)
        print(trade.orderStatus.status)
        ib.sleep(5)
        if trade.orderStatus.status == 'Filled' or trade.orderStatus.status == 'Inactive':
            print('sell loop finished')
            return
        else:
            ib.cancelOrder(order)
            return


def option_position():
    stock_owned = np.zeros(2)
    position = ib.portfolio()
    call_position = None
    put_position = None
    for each in position:
        if each.contract.right == 'C':
            call_position = each.contract
            ib.qualifyContracts(call_position)
            stock_owned[0] = each.position
        elif each.contract.right == 'P':
            put_position = each.contract
            ib.qualifyContracts(put_position)
            stock_owned[1] = each.position
    call_position = call_position if call_position != None else res.get_contract('C', 2000)
    put_position = put_position if put_position != None else res.get_contract('P', 2000)
    return stock_owned, call_position, put_position


def trade(ES, hasNewBar=None, p=-1):
    global data
    global call_option_price
    global put_option_price
    global buy_index
    global sell_index
    global call_option_volume
    global put_option_volume
    global account

    buy_index = []
    sell_index = []
    tickers_signal = "Hold"
    stock_owned, call_contract, put_contract = option_position()
    cash_in_hand = float(account[22].value)
    portolio_value = float(account[29].value)

    call_contract_price = (call_option_price.ask + call_option_price.bid) / 2
    put_contract_price = (put_option_price.ask + put_option_price.bid) / 2
    options_array = np.array([call_contract_price, put_contract_price])
    call_option_volume = roll_contract(call_option_volume, call_option_price.bidSize)
    put_option_volume = roll_contract(put_option_volume, put_option_price.bidSize)
    # options_bid_volume = np.array([call_option_volume,put_option_volume])
    data_raw = res.ES(ES)
    open_orders = len(ib.reqAllOpenOrders())
    holding_position = len(ib.portfolio())

    df = data_raw[['high', 'low', 'volume', 'close', 'RSI', 'ATR', 'roll_max_cp', 'roll_min_cp', 'roll_max_vol', 'macd',
                   'macdsignal']].tail()

    # print(
    #     f'cash in hand = {cash_in_hand}, portfolio value = {portolio_value}, unrealized PNL = {account[32].value}, realized PNL = {account[33].value}, holding = {events.stock_owned[0]} calls and {events.stock_owned[1]} puts and ES = {data_raw.iloc[-1, 3]} and [call,puts] values are = {options_array}')

    print(df["high"].iloc[-1], df["low"].iloc[-1], df["roll_max_cp"].iloc[-2], df["roll_max_cp"].iloc[-2],
          df["volume"].iloc[-1], df["roll_max_vol"].iloc[-2], df["close"].iloc[-1],
          df["close"].iloc[-2] - (1.5 * df["ATR"].iloc[-2]), call_option_volume[-1], call_option_volume.max() / 2)
    if df["high"].iloc[-1] >= df["roll_max_cp"].iloc[-2] and \
            df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and \
            holding_position == 0 and buy_index == []:

        tickers_signal = "Buy call"
        buy_index.append(0)

    elif df["low"].iloc[-1] <= df["roll_min_cp"].iloc[-2] and \
            df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and \
            holding_position == 0 and buy_index == []:

        tickers_signal = "Buy put"
        buy_index.append(1)

    elif df["low"].iloc[-1] <= df["roll_min_cp"].iloc[-2] and \
            df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and df['RSI'].iloc[-1] < 70 \
            and holding_position != 0 and open_orders == 0 and sell_index == [] and buy_index == []:
        tickers_signal = "sell call and buy puts"
        sell_index.append(0)
        buy_index.append(1)


    elif df["high"].iloc[-1] >= df["roll_max_cp"].iloc[-2] and \
            df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and df['RSI'].iloc[-1] > 30 \
            and holding_position != 0 and open_orders == 0 and sell_index == [] and buy_index == []:
        tickers_signal = "sell put and buy calls"
        sell_index.append(1)
        buy_index.append(0)

    elif (df["close"].iloc[-1] < df["close"].iloc[-2] - (1.5 * df["ATR"].iloc[-2]) \
          or (df["close"].iloc[-1] < df["low"].iloc[-2] and \
              df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2]) or \
          call_option_volume[-1] <= call_option_volume.max() / 2) and \
            holding_position != 0 and open_orders == 0 and \
            sell_index == [] and buy_index == []:
        print('1 ************************')
        tickers_signal = "sell call"
        sell_index.append(0)

    elif (df["close"].iloc[-1] > df["close"].iloc[-2] + (1.5 * df["ATR"].iloc[-2]) \
          or (df["close"].iloc[-1] > df["high"].iloc[-2] and \
              df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2]) or \
          put_option_volume[-1] <= put_option_volume.max() / 2) and \
            holding_position != 0 and open_orders == 0 and sell_index == [] and buy_index == []:
        print('2 ************************')
        tickers_signal = "sell put"
        sell_index.append(1)

    print(tickers_signal)

    if sell_index:

        for i in sell_index:
            if ((stock_owned[0] != 0 and i == 0) or (stock_owned[1] != 0 and i == 1)) and open_orders == 0 and not holding_position == 0:
                contract = call_contract if i == 0 else put_contract
                ib.qualifyContracts(contract)
                price = call_option_price if i == 0 else put_option_price
                flatten_position(contract, price)
                ib.sleep(0)
                cash_in_hand = float(ib.accountSummary()[5].value)
                stock_owned, call_contract, put_contract = option_position()
                sell_index = []

    if buy_index:

        for i in buy_index:
            contract = call_contract if i == 0 else put_contract
            ib.qualifyContracts(contract)

            if cash_in_hand > (options_array[i] * 50) and cash_in_hand > portolio_value \
                    and ((stock_owned[0] == 0 and i == 0) or (stock_owned[1] == 0 and i == 1)) and holding_position == 0 and open_orders == 0:
                options_array[i] = call_option_price.ask + 0.25 if i == 0 else put_option_price.ask + 0.25

                quantity = 1  # int((cash_in_hand/(options_array[i] * 50)))

                order = LimitOrder('BUY', quantity, options_array[i])  # round(25 * round(options_array[i]/25, 2), 2))
                trade = ib.placeOrder(contract, order)
                print(f'buying {"CALL" if contract.right == "C" else "PUT"}')
                print(trade.orderStatus.status)
                stock_owned, call_contract, put_contract = option_position()

                buy_index = []
            else:
                buy_index = []


def show_error(reqId, errorCode, errorString, contract):
    print(contract, errorCode, errorString)
    ib.client.reset()
    ib.client.MaxRequests = 0
    ib.sleep(0)


def connect():
    if not ib.isConnected():
        ib.connect('127.0.0.1', 7497, clientId=np.random.randint(10, 1000))
        ib.sleep(0)
        ib.client.MaxRequests = 45


def roll_contract(option_vol, value):
    option_vol = np.roll(option_vol, -1)
    option_vol[-1] = value
    return option_vol


def account_update(value):
    global account
    account = ib.accountSummary()
    return account


if __name__ == "__main__":
    global call_option_price
    global put_option_price
    global call_option_volume
    global put_option_volume
    import talib as ta
    from talib import MA_Type

    events = all_events()
    global account
    ib = IB()
    ib.disconnect()
    connect()
    ib.accountSummaryEvent += account_update
    ib.errorEvent += show_error
    ib.disconnectedEvent += events.disconnected
    ib.updatePortfolioEvent += events.portfolio_update

    call_option_volume = np.ones(20)
    put_option_volume = np.ones(20)
    events.stock_owned = np.zeros(2)
    endDateTime = ''
    No_days = '2 D'
    interval = '1 min'
    res = get_data()
    ES = Future(symbol='ES', lastTradeDateOrContractMonth='20201218', exchange='GLOBEX',
                currency='USD')
    ib.qualifyContracts(ES)
    ES = ib.reqHistoricalData(contract=ES, endDateTime='', durationStr=No_days,
                              barSizeSetting=interval, whatToShow='TRADES', useRTH=False, keepUpToDate=True,
                              timeout=10)
    events.portfolio_update()
    # events.stock_owned, events.call_position, events.put_position = events.portfolio_update()
    ib.sleep(0)

    call_option_price = ib.reqMktData(
        events.call_position if events.call_position != None else res.get_contract('C', 2000), '', False, False)
    put_option_price = ib.reqMktData(
        events.put_position if events.put_position != None else res.get_contract('P', 2000), '', False, False)

    call_option_volume = roll_contract(call_option_volume, call_option_price.bidSize)
    put_option_volume = roll_contract(put_option_volume, put_option_price.bidSize)
    account = account_update(0)

    trade(ES)

    ES.updateEvent += trade
    ib.run()



