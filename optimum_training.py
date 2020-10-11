import numpy as np
import pandas as pd
import talib as ta
from talib import MA_Type
from ib_insync import *
import asyncio
from datetime import datetime, timedelta
from ressup import ressup
import nest_asyncio
nest_asyncio.apply()
ib = IB()
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
        ES_df['macd'], ES_df['macdsignal'], ES_df['macdhist'] = ta.MACD(ES_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
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

    def option_history(self, contract):
        df = pd.DataFrame(util.df(ib.reqHistoricalData(contract=contract, endDateTime='', durationStr='3 D',
                                      barSizeSetting='1 min', whatToShow = 'MIDPOINT', useRTH = False, keepUpToDate=False))[['date','close']])
        df.columns=['date',f"{contract.symbol}_{contract.right}_close"]
        df.set_index('date',inplace=True)
        return df

    def options(self, df1,df2):
        return pd.merge(df1,df2, on='date', how='outer').dropna()



class Trade:


    def __init__(self):
        self.connect()
        self.call_option_volume = np.ones(20)
        self.put_option_volume = np.ones(20)
        self.stock_owned = np.zeros(2)

        ES = Future(symbol='ES', lastTradeDateOrContractMonth='20201218', exchange='GLOBEX',
                    currency='USD')
        ib.qualifyContracts(ES)
        self.ES = ib.reqHistoricalData(contract=ES, endDateTime='', durationStr='2 D',
                                       barSizeSetting='1 min', whatToShow='TRADES', useRTH=False, keepUpToDate=True,
                                       timeout=10)
        self.option_position()
        self.call_option_price = ib.reqMktData(self.call_contract, '', False, False)
        self.put_option_price = ib.reqMktData(self.put_contract, '', False, False)
        self.call_option_volume = self.roll_contract(self.call_option_volume, self.call_option_price.bidSize)
        self.put_option_volume = self.roll_contract(self.put_option_volume, self.put_option_price.bidSize)
        self.submitted = 0
        for each in ib.positions():
            if each.position < 0:
                order = MarketOrder(action="BUY", totalQuantity=abs(each.position))
                ib.qualifyContracts(each.contract)
                trade = ib.placeOrder(contract=each.contract, order=order)
                print(f'Buying to close short position')
        self.ATR = 1
        self.ATR_checking = 0

        def profitable_ATR(self):
            print('checking ATR and max_vol roll period')
            ES = Future(symbol='ES', lastTradeDateOrContractMonth='20201218', exchange='GLOBEX',
                        currency='USD')
            ib.qualifyContracts(ES)

            ES = ib.reqHistoricalData(contract=ES, endDateTime='', durationStr='2 D',
                                       barSizeSetting='1 min', whatToShow='TRADES', useRTH=False, keepUpToDate=True,
                                       timeout=10) # start data collection for ES-Mini
            data_raw = res.options(res.options(res.ES(ES), res.option_history(res.get_contract('C', 2000))),
                                    res.option_history(res.get_contract('P', 2000)))
            df = data_raw[
                ['high', 'low', 'close', 'volume', 'roll_max_vol', 'roll_max_cp', 'roll_min_cp', 'ATR', 'ES_C_close',
                 'ES_P_close']]
                        var_ret = {}
            var = 0.25
            buy_index = []
            sell_index = []
            stock_owned = np.zeros(2)
            var_ret = {}
            while var != 5:
                ret = []
                for i in range(len(df)):
                    if df["high"].iloc[i] >= df["roll_max_cp"].iloc[i - 1] and \
                            df["volume"].iloc[i] > df["roll_max_vol"].iloc[i - 1] and \
                            buy_index == [] and stock_owned[0] == 0 and stock_owned[1] == 0:

                        tickers_signal = "Buy call"
                        buy_index.append(0)
                        stock_owned = [1, 0]

                    elif df["low"].iloc[i] <= df["roll_min_cp"].iloc[i - 1] and \
                            df["volume"].iloc[i] > df["roll_max_vol"].iloc[i - 1] and \
                            buy_index == [] and stock_owned[0] == 0 and stock_owned[1] == 0:

                        tickers_signal = "Buy put"
                        buy_index.append(1)
                        stock_owned = [0, 1]

                    elif df["low"].iloc[i] <= df["roll_min_cp"].iloc[i - 1] and df["volume"].iloc[i] > \
                            df["roll_max_vol"].iloc[i - 1] and stock_owned[0] == 1 and stock_owned[1] == 0:
                        tickers_signal = "sell call and buy puts"
                        sell_index.append(0)
                        buy_index.append(1)
                        stock_owned = [0, 1]


                    elif df["high"].iloc[i] >= df["roll_max_cp"].iloc[i - 1] and df["volume"].iloc[i] > \
                            df["roll_max_vol"].iloc[i - 1] and stock_owned[1] == 1 and stock_owned[0] == 0:
                        tickers_signal = "sell put and buy calls"
                        sell_index.append(1)
                        buy_index.append(0)
                        stock_owned = [1, 0]

                    elif (df["close"].iloc[i] < df["close"].iloc[i - 1] - (var * df["ATR"].iloc[i - 1]) or (
                            (df["close"].iloc[i] < df["low"].iloc[i - 1]) and df["volume"].iloc[i] >
                            df["roll_max_vol"].iloc[i - 1])) and stock_owned[0] == 1 and stock_owned[1] == 0:
                        tickers_signal = "sell call"
                        sell_index.append(0)
                        stock_owned = [0, 0]


                    elif (df["close"].iloc[i] > df["close"].iloc[i - 1] + (var * df["ATR"].iloc[i - 1]) or (
                            (df["close"].iloc[i] > df["high"].iloc[i - 1]) and df["volume"].iloc[i] >
                            df["roll_max_vol"].iloc[i - 1])) and stock_owned[0] == 0 and stock_owned[1] == 1:
                        tickers_signal = "sell put"
                        sell_index.append(1)
                        stock_owned = [0, 0]
                    else:
                        tickers_signal = "Hold"
                        sell_index = []
                        buy_index = []
                        if stock_owned[0] != 0:
                            ret.append(df['ES_C_close'][i] - df['ES_C_close'][i - 1])
                        elif stock_owned[1] != 0:
                            ret.append(df['ES_P_close'][i] - df['ES_P_close'][i - 1])
                        else:
                            ret.append(0)

                print(f'var = {var}, returns = {sum(ret) * 50}')
                var_ret[str(var)] = sum(ret) * 50
                var += 0.25

            ATR = max(var_ret, key=var_ret.get)
            print('Max ATR is at', ATR)
            vol_var = 5
            var_ret = {}
            while vol_var <= 50:
                df['roll_max_vol'] = df['volume'].rolling(vol_var).max()
                buy_index = []
                sell_index = []
                stock_owned = np.zeros(2)

                ret = []
                for i in range(len(df)):
                    if df["high"].iloc[i] >= df["roll_max_cp"].iloc[i - 1] and \
                            df["volume"].iloc[i] > df["roll_max_vol"].iloc[i - 1] and \
                            buy_index == [] and stock_owned[0] == 0 and stock_owned[1] == 0:

                        tickers_signal = "Buy call"
                        buy_index.append(0)
                        stock_owned = [1, 0]

                    elif df["low"].iloc[i] <= df["roll_min_cp"].iloc[i - 1] and \
                            df["volume"].iloc[i] > df["roll_max_vol"].iloc[i - 1] and \
                            buy_index == [] and stock_owned[0] == 0 and stock_owned[1] == 0:

                        tickers_signal = "Buy put"
                        buy_index.append(1)
                        stock_owned = [0, 1]

                    elif df["low"].iloc[i] <= df["roll_min_cp"].iloc[i - 1] and df["volume"].iloc[i] > \
                            df["roll_max_vol"].iloc[i - 1] and stock_owned[0] == 1 and stock_owned[1] == 0:
                        tickers_signal = "sell call and buy puts"
                        sell_index.append(0)
                        buy_index.append(1)
                        stock_owned = [0, 1]


                    elif df["high"].iloc[i] >= df["roll_max_cp"].iloc[i - 1] and df["volume"].iloc[i] > \
                            df["roll_max_vol"].iloc[i - 1] and stock_owned[1] == 1 and stock_owned[0] == 0:
                        tickers_signal = "sell put and buy calls"
                        sell_index.append(1)
                        buy_index.append(0)
                        stock_owned = [1, 0]

                    elif (df["close"].iloc[i] < df["close"].iloc[i - 1] - (float(ATR) * df["ATR"].iloc[i - 1]) or (
                            (df["close"].iloc[i] < df["low"].iloc[i - 1]) and df["volume"].iloc[i] >
                            df["roll_max_vol"].iloc[i - 1])) and stock_owned[0] == 1 and stock_owned[1] == 0:
                        tickers_signal = "sell call"
                        sell_index.append(0)
                        stock_owned = [0, 0]


                    elif (df["close"].iloc[i] > df["close"].iloc[i - 1] + (float(ATR) * df["ATR"].iloc[i - 1]) or (
                            (df["close"].iloc[i] > df["high"].iloc[i - 1]) and df["volume"].iloc[i] >
                            df["roll_max_vol"].iloc[i - 1])) and stock_owned[0] == 0 and stock_owned[1] == 1:
                        tickers_signal = "sell put"
                        sell_index.append(1)
                        stock_owned = [0, 0]
                    else:
                        tickers_signal = "Hold"
                        sell_index = []
                        buy_index = []
                        if stock_owned[0] != 0:
                            ret.append(df['ES_C_close'][i] - df['ES_C_close'][i - 1])
                        elif stock_owned[1] != 0:
                            ret.append(df['ES_P_close'][i] - df['ES_P_close'][i - 1])
                        else:
                            ret.append(0)

                print(f'vol_var = {vol_var}, returns = {sum(ret) * 50}')
                var_ret[str(vol_var)] = sum(ret) * 50
                vol_var += 5

            vol = max(var_ret, key=var_ret.get)
            print('Max roll var is at', vol)
            return ATR, vol

    def connect(self):
        ib.disconnect()
        ib.connect('127.0.0.1', 7497, clientId=np.random.randint(10, 1000))
        ib.client.MaxRequests = 55

endDateTime = ''
No_days = '1 D'
interval = '1 min'
trade = Trade()

trade.connect()
res = get_data()

data_raw = res.options(res.options(res.ES(),res.option_history(res.get_contract('C', 2000)))\
                               ,res.option_history(res.get_contract('P', 2000)))