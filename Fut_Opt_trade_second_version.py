# -*- coding: utf-8 -*-
"""
Created on Mon May 18 23:43:59 2020

@author: alial
"""

import numpy as np
from ib_insync import IB, MarketOrder, util, Future, FuturesOption
import datetime
import talib as ta
import nest_asyncio

nest_asyncio.apply()


class trade_ES():
    def __init__(self):

        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=np.random.randint(10, 1000))
        self.tickers_ret = {}
        self.endDateTime = ''
        self.No_days = '43200 S'
        self.interval = '30 secs'
        self.tickers_signal = "Hold"
        self.ES = Future(symbol='ES', lastTradeDateOrContractMonth='20200619', exchange='GLOBEX',
                         currency='USD')
        self.ib.qualifyContracts(self.ES)
        self.ES_df = self.ib.reqHistoricalData(contract=self.ES, endDateTime=self.endDateTime, durationStr=self.No_days,
                                          barSizeSetting=self.interval, whatToShow='TRADES', useRTH=False,
                                          keepUpToDate=True)
        self.tickers_ret = []
        self.options_ret = []
        self.option = {'call': FuturesOption, 'put': FuturesOption}
        self.options_history = {}
        self.trade_options = {'call': [], 'put': []}
        self.price = 0
        self.i = -1
        self.ES_df.updateEvent += self.make_clean_df
        self.Buy = True
        self.Sell = False
        self.ib.positionEvent += self.order_verify
        self.waitTimeInSeconds = 220
        self.tradeTime = 0

    def run(self):

        while self.ib.waitOnUpdate():
            util.allowCtrlC()
            self.ib.setCallback('error', x.checkError)
            self.make_clean_df(self.ES_df)

    def next_exp_weekday(self):

        weekdays = {2: [6, 0], 4: [0, 1, 2], 0: [3, 4]}
        today = datetime.date.today().weekday()
        for exp, day in weekdays.items():
            if today in day:
                return exp

    def next_weekday(self, d, weekday):

        days_ahead = weekday - d.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        date_to_return = d + datetime.timedelta(days_ahead)  # 0 = Monday, 1=Tuself.ESday, 2=Wednself.ESday...
        return date_to_return.strftime('%Y%m%d')

    def get_strikes_and_expiration(self):

        expiration = self.next_weekday(datetime.date.today(), self.next_exp_weekday())
        chains = self.ib.reqSecDefOptParams(underlyingSymbol='ES', futFopExchange='GLOBEX', underlyingSecType='FUT',
                                            underlyingConId=self.ES.conId)
        chain = util.df(chains)
        strikes = chain[chain['expirations'].astype(str).str.contains(expiration)].loc[:, 'strikes'].values[0]
        [ESValue] = self.ib.reqTickers(self.ES)
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
            self.ib.qualifyContracts(contract)
            self.price = self.ib.reqMktData(contract, "", False, False)
            if float(self.price.last) * 50 >= net_liquidation:
                continue
            else:
                return contract

    def make_clean_df(self, ES_df, hashbar=None):

        ES_df = util.df(ES_df)
        ES_df['RSI'] = ta.RSI(ES_df['close'])
        ES_df['macd'], ES_df['macdsignal'], ES_df['macdhist'] = ta.MACD(ES_df['close'], fastperiod=12, slowperiod=26,
                                                                        signalperiod=9)
        ES_df['MA_9'] = ta.MA(ES_df['close'], timeperiod=9)
        ES_df['MA_21'] = ta.MA(ES_df['close'], timeperiod=21)
        ES_df['MA_200'] = ta.MA(ES_df['close'], timeperiod=200)
        ES_df['EMA_9'] = ta.EMA(ES_df['close'], timeperiod=9)
        ES_df['EMA_21'] = ta.EMA(ES_df['close'], timeperiod=21)
        ES_df['EMA_200'] = ta.EMA(ES_df['close'], timeperiod=200)
        ES_df['ATR'] = ta.ATR(ES_df['high'], ES_df['low'], ES_df['close'])
        ES_df['roll_max_cp'] = ES_df['high'].rolling(20).max()
        ES_df['roll_min_cp'] = ES_df['low'].rolling(20).min()
        ES_df['roll_max_vol'] = ES_df['volume'].rolling(20).max()
        ES_df.dropna(inplace=True)
        self.loop_function(ES_df)
        
    def placeOrder(self, contract, order):
        
        trade = self.ib.placeOrder(contract, order)
        tradeTime = datetime.datetime.now()
        return([trade, contract, tradeTime])

    def sell(self, contract, position):
        
        self.ib.qualifyContracts(contract)
        if position.position>0:
            order = 'Sell'
        else:
            order = 'Buy'

        marketorder = MarketOrder(order, abs(position.position))

        if self.tradeTime!=0:
            timeDelta = datetime.datetime.now() - self.tradeTime
            if timeDelta.seconds > self.waitTimeInSeconds:
                marketTrade, contract, self.tradeTime = self.placeOrder(contract, marketorder)
                print(f'self.tradeTime = {self.tradeTime}, timeDelta = {timeDelta}' )
        else:
            marketTrade, contract, tradeTime = self.placeOrder(contract, marketorder)
            print(f'zero / self.tradeTime = {self.tradeTime}, timeDelta = {timeDelta}' )
        condition = marketTrade.isDone
        timeout = 20
        for c in self.ib.loopUntil(condition=condition, timeout=timeout):
            marketorder = MarketOrder('Sell', position.position)
            marketTrade = self.ib.placeOrder(contract, marketorder)

        if not condition == 'Filled':
            self.ib.cancelOrder(marketorder)
            marketorder = MarketOrder('Sell', position.position)
            marketTrade = self.ib.placeOrder(contract, marketorder)
            
    def buy(self, contract):
        self.ib.qualifyContracts(contract)
        marketorder = MarketOrder('Buy', 1)
        if self.tradeTime!=0:
            timeDelta = datetime.datetime.now() - self.tradeTime
            if timeDelta.seconds > self.waitTimeInSeconds:
                marketTrade, contract, self.tradeTime = self.placeOrder(contract, marketorder)
                print(f'self.tradeTime = {self.tradeTime}, timeDelta = {timeDelta}' )
        else:
            marketTrade, contract, tradeTime = self.placeOrder(contract, marketorder)
            print(f'zero / self.tradeTime = {self.tradeTime}, timeDelta = {timeDelta}' )
        condition = marketTrade.isDone
        timeout = 10
        for c in self.ib.loopUntil(condition=condition, timeout=timeout):
            marketorder = MarketOrder('Buy', 1)
            marketTrade = self.ib.placeOrder(contract, marketorder)
        if not condition == 'Filled':
            self.ib.cancelOrder(marketorder)
            marketorder = MarketOrder('Buy', 1)
            marketTrade = self.ib.placeOrder(contract, marketorder)

    def order_verify(self, order):
        if order.position == 0.0 or order.position < 0:
            self.Buy= True
            self.Sell= False
        elif order.position > 0:
            self.Buy = False
            self.Sell = True
            
        else:
            self.Buy = False
            self.Sell = False
        print(f'Buy= {self.Buy}, sell = {self.Sell}')


    def loop_function(self, ES_df):

        df = ES_df[
            ['high', 'low', 'volume', 'close', 'RSI', 'ATR', 'roll_max_cp', 'roll_min_cp', 'roll_max_vol', 'EMA_9',
             'EMA_21', 'macd', 'macdsignal']]

        if self.tickers_signal == "Hold":
            print('Hold')
            if df["high"].iloc[self.i] >= df["roll_max_cp"].iloc[self.i] and \
                    df["volume"].iloc[self.i] > df["roll_max_vol"].iloc[self.i - 1] and df['RSI'].iloc[self.i] > 30 \
                    and df['macd'].iloc[self.i] > df['macdsignal'].iloc[self.i] :
                self.tickers_signal = "Buy"
                return

                
            elif df["low"].iloc[self.i] <= df["roll_min_cp"].iloc[self.i] and \
                    df["volume"].iloc[self.i] > df["roll_max_vol"].iloc[self.i - 1] and df['RSI'].iloc[self.i] < 70 \
                    and df['macd'].iloc[self.i] < df['macdsignal'].iloc[self.i]:
                self.tickers_signal = "Sell"
                return
            
            else:
                self.tickers_signal = "Hold"
                return


        elif self.tickers_signal == "Buy":
            print('BUY SIGNAL')
            if df["close"].iloc[self.i] > df["close"].iloc[self.i - 1] - (0.75 * df["ATR"].iloc[self.i - 1]) and len(self.ib.positions())!=0:
                print(f'{df["close"].iloc[self.i]} > {df["close"].iloc[self.i - 1] - (0.75 * df["ATR"].iloc[self.i - 1])}')
                print('first buy condition')
                positions = self.ib.positions()
                for position in positions:
                    if position.contract.right == 'C':
                        self.sell(position.contract, position)
                        self.tickers_signal = "Hold"
                        return
                


            elif df["low"].iloc[self.i] <= df["roll_min_cp"].iloc[self.i] and \
                    df["volume"].iloc[self.i] > df["roll_max_vol"].iloc[self.i - 1] and df['RSI'].iloc[self.i] < 70 \
                    and df['macd'].iloc[self.i] < df['macdsignal'].iloc[self.i] and len(self.ib.positions())!=0:
                self.tickers_signal = "Sell"
                print('sell')
                positions = self.ib.positions()
                for position in positions:
                    if position.contract.right == 'P':
                        self.sell(position.contract, position)
                        self.tickers_signal == "Buy"
                        return
                

            else:
                if len(self.ib.positions())==0:            
                    self.option['call'] = self.get_contract(right="C", net_liquidation=2000)
                    self.buy(self.option['call'])
                    self.tickers_signal = "Hold"
                else:
                    self.tickers_signal = "Hold"


        elif self.tickers_signal == "Sell":
            print('SELL SIGNAL')
            if df["close"].iloc[self.i] < df["close"].iloc[self.i - 1] + (0.75 * df["ATR"].iloc[self.i - 1]) and len(self.ib.positions())!=0:
                print('first sell condition')
                print(f'{df["close"].iloc[self.i]} < {df["close"].iloc[self.i - 1] - (0.75 * df["ATR"].iloc[self.i - 1])}')
                print('sell')
                positions = self.ib.positions()
                for position in positions:
                    if position.contract.right == 'P':
                        self.sell(position.contract, position)
                        self.tickers_signal = "Hold"
                        return
            


            elif df["high"].iloc[self.i] >= df["roll_max_cp"].iloc[self.i] and \
                    df["volume"].iloc[self.i] > df["roll_max_vol"].iloc[self.i - 1] and df['RSI'].iloc[self.i] > 30 \
                    and df['macd'].iloc[self.i] > df['macdsignal'].iloc[self.i] and len(self.ib.positions())!=0:
                self.tickers_signal = "Buy"
                print('sell')
                positions = self.ib.positions()
                for position in positions:
                    if position.contract.right == 'P':
                        self.sell(position.contract, position)
                        self.tickers_signal == "Sell"
                        return

            else:
                if len(self.ib.positions())==0:   
                    self.option['put'] = self.get_contract(right="P", net_liquidation=2000)
                    self.buy(self.option['put'])
                    self.tickers_signal = "Hold"
                else:
                    self.tickers_signal = "Hold"



    def checkError(self, errCode, errString):
        print('Error Callback', errCode, errString)
        if errCode == 2104:
            print('re-connect after 5 secs')
            self.ib.sleep(5)
            self.ib.disconnect()
            self.ib.connect('127.0.0.1', 7497, clientId=np.random.randint(10, 1000))
            self.make_clean_df(self.ES)



if __name__ == '__main__':
    ib=IB()
    x = trade_ES()
    try:
        while ib.waitOnUpdate():
            x.run()
    except Exception as error:
        print(error)