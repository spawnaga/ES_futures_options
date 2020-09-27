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

res = get_data()

data_raw = res.options(res.options(res.ES(),res.option_history(res.get_contract('C', 2000)))\
                               ,res.option_history(res.get_contract('P', 2000))) #collect live data of ES with TA and options prices

#
# class trade:
#     def trade(self, ES):
#         buy_index = []
#         sell_index = []
#         tickers_signal = "Hold"
#         account = ib.accountSummary()
#         cash_in_hand = float(account[22].value)
#         portolio_value = float(account[29].value)
#         portfolio = ib.portfolio()
#         open_orders = ib.reqAllOpenOrders()
#
#         self.call_contract_price = 0.25 * round(((self.call_option_price.ask + self.call_option_price.bid) / 2) / 0.25)
#         self.put_contract_price = 0.25 * round(((self.put_option_price.ask + self.put_option_price.bid) / 2) / 0.25)
#         options_array = np.array([self.call_contract_price, self.put_contract_price])
#         self.call_option_volume = self.roll_contract(self.call_option_volume, self.call_option_price.bidSize)
#         self.put_option_volume = self.roll_contract(self.put_option_volume, self.put_option_price.bidSize)
#         # options_bid_volume = np.array([self.call_option_volume,self.put_option_volume])
#         data_raw = res.ES(ES)
#
#         df = data_raw[
#             ['high', 'low', 'volume', 'close', 'RSI', 'ATR', 'roll_max_cp', 'roll_min_cp', 'roll_max_vol']].tail()
#
#         print(
#             f'cash in hand = {cash_in_hand}, portfolio value = {portolio_value}, unrealized PNL = {account[32].value}, realized PNL = {account[33].value}, holding = {self.stock_owned[0]} calls and {self.stock_owned[1]} puts and ES = {data_raw.iloc[-1, 3]} and [call,puts] values are = {options_array}')
#         if df["high"].iloc[-1] >= df["roll_max_cp"].iloc[-2] and \
#                 df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and \
#                 len(portfolio) == 0 and buy_index == []:
#
#             tickers_signal = "Buy call"
#             buy_index.append(0)
#
#         elif df["low"].iloc[-1] <= df["roll_min_cp"].iloc[-2] and \
#                 df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and \
#                 len(portfolio) == 0 and buy_index == []:
#
#             tickers_signal = "Buy put"
#             buy_index.append(1)
#
#         elif df["low"].iloc[-1] <= df["roll_min_cp"].iloc[-2] and \
#                 df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and df['RSI'].iloc[-1] < 70 \
#                 and len(portfolio) != 0 and len(
#             open_orders) == 0 and sell_index == [] and buy_index == []:
#             tickers_signal = "sell call and buy puts"
#             sell_index.append(0)
#             buy_index.append(1)
#
#
#         elif df["high"].iloc[-1] >= df["roll_max_cp"].iloc[-2] and \
#                 df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and df['RSI'].iloc[-1] > 30 \
#                 and len(portfolio) != 0 and len(
#             open_orders) == 0 and sell_index == [] and buy_index == []:
#             tickers_signal = "sell put and buy calls"
#             sell_index.append(1)
#             buy_index.append(0)
#
#         elif (df["close"].iloc[-1] < df["close"].iloc[-2] - (1.5 * df["ATR"].iloc[-2]) \
#               or (df["close"].iloc[-1] < df["low"].iloc[-2] and \
#                   df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2]) or \
#               2 <= self.call_option_volume[-1] <= self.call_option_volume.max() / 4) and \
#                 self.stock_owned[0] != 0 and len(portfolio) != 0 and len(open_orders) == 0 and \
#                 sell_index == [] and buy_index == []:
#             print('1 ************************')
#             tickers_signal = "sell call"
#             sell_index.append(0)
#
#         elif (df["close"].iloc[-1] > df["close"].iloc[-2] + (1.5 * df["ATR"].iloc[-2]) \
#               or (df["close"].iloc[-1] > df["high"].iloc[-2] and \
#                   df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2]) or \
#               2 <= self.put_option_volume[-1] <= self.put_option_volume.max() / 4) and \
#                 self.stock_owned[1] != 0 and len(portfolio) != 0 and len(
#             open_orders) == 0 and sell_index == [] and buy_index == []:
#             print('2 ************************')
#             tickers_signal = "sell put"
#             sell_index.append(1)
#
#         print(tickers_signal)
#         print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
#
#         if sell_index:
#
#             for i in sell_index:
#                 print(self.stock_owned[i])
#                 print(len(portfolio))
#
#                 if len(portfolio) != 0 and len(open_orders) == 0 and self.submitted == 0:
#                     self.submitted = 1
#                     contract = self.call_contract if i == 0 else self.put_contract
#                     ib.qualifyContracts(contract)
#                     price = self.call_option_price if i == 0 else self.put_option_price
#                     self.flatten_position(contract, price)
#             sell_index = []
#
#         if buy_index:
#
#             for i in buy_index:
#                 contract = self.call_contract if i == 0 else self.put_contract
#                 ib.qualifyContracts(contract)
#
#                 if cash_in_hand > (options_array[i] * 50) and cash_in_hand > portolio_value \
#                         and ((self.stock_owned[0] == 0 and i == 0) or (self.stock_owned[1] == 0 and i == 1)) and len(
#                     open_orders) == 0 and len(ib.positions()) == 0:
#                     options_array[i] = self.call_option_price.ask + 0.25 if i == 0 else self.put_option_price.ask + 0.25
#                     quantity = 1  # int((cash_in_hand/(options_array[i] * 50)))
#                     self.open_position(contract=contract, quantity=quantity, options_array=options_array[i])
#             buy_index = []