from datetime import datetime, timedelta, time

import nest_asyncio
import numpy as np
import pandas as pd
import statsmodels.api as sm
import talib as ta
from ib_insync import *
from stocktrends import Renko
import sys
import math
nest_asyncio.apply()  # enable nest asyncio
sys.setrecursionlimit(10 ** 9)  # set recursion limit to 1000000000
pd.options.mode.chained_assignment = None  # remove a warning

def x_round(x):
    return round(x*4)/4

class get_data:
    """ A class to get ES Technical analysis and next 2 days expiration date and delta 60 option strikes for
    whatever ES price at """

    def __init__(self):
        pass

    def next_exp_weekday(self):
        """ Set next expiration date for contract 0 = Monday, 1 = Tuesday, etc..."""
        weekdays = {2: [5, 6, 0], 4: [0, 1, 2], 0: [3, 4]}
        today = datetime.today().weekday()
        for exp, day in weekdays.items():
            if today in day:
                return exp  # return the 2nd next weekday number

    def next_weekday(self, d, weekday):
        """ Translate weekdays number to a date for example next Mon = October 19th 2020"""
        days_ahead = weekday - d.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        date_to_return = d + timedelta(days_ahead)  # 0 = Monday, 1=Tus self.ES day, 2=Wed self.ES day...
        return date_to_return.strftime('%Y%m%d')  # return the date in the form of (yearmonthday) ex:(20201019)

    def get_strikes_and_expiration(self):
        """ When used, returns strikes and expiration for the ES futures options"""
        ES = Future(symbol='ES', lastTradeDateOrContractMonth='20210917', exchange='GLOBEX',
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
        """ Get contracts for ES futures options by using get_strikes_and_expiration function"""
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

    def slope(self, ser, n):
        """function to calculate the slope of n consecutive points on a plot"""
        slopes = [i * 0 for i in range(n - 1)]
        for i in range(n, len(ser) + 1):
            y = ser[i - n:i]
            x = np.array(range(n))
            y_scaled = (y - y.min()) / (y.max() - y.min())
            x_scaled = (x - x.min()) / (x.max() - x.min())
            x_scaled = sm.add_constant(x_scaled)
            model = sm.OLS(y_scaled, x_scaled)
            results = model.fit()
            slopes.append(results.params[-1])
        slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
        return np.array(slope_angle)

    def renko_df(self, df_raw, ATR=120):
        # df_raw = df_raw[-500:]
        # df_raw.reset_index(inplace=True)
        df_raw = df_raw.reset_index()
        renko = Renko(df_raw[['date', 'open', 'high', 'low', 'close', 'volume']])
        renko.brick_size = ATR

        df = renko.get_ohlc_data()
        df['bar_num'] = np.where(df['uptrend'] == True, 1, np.where(df['uptrend'] == False, -1, 0))

        for i in range(1, len(df["bar_num"])):
            if df["bar_num"].iloc[i] > 0 and df["bar_num"].iloc[i - 1] > 0:
                df["bar_num"].iloc[i] += df["bar_num"].iloc[i - 1]
            elif df["bar_num"].iloc[i] < 0 and df["bar_num"].iloc[i - 1] < 0:
                df["bar_num"].iloc[i] += df["bar_num"].iloc[i - 1]
        df.drop_duplicates(subset="date", keep="last", inplace=True)
        df_raw = df_raw.merge(df.loc[:, ["date", "bar_num"]], how="outer", on="date")
        df_raw["bar_num"].fillna(method='ffill', inplace=True)
        # df_raw["adx_slope"] = slope(df_raw['adx'], 5)
        # print(df_raw.iloc[:2,:])
        # print(f'**************{len(df_raw)}**********************')
        return df_raw

    def tech_analysis(self, df, period):
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], 10)
        df = df.reset_index().fillna(method='ffill')
        df = self.renko_df(df, df['atr'].mean())

        df['OBV'] = ta.OBV(df['close'], df['volume'])
        df["obv_slope"] = self.slope(df['OBV'], 5)
        df["roll_max_cp"] = df["high"].rolling(10).max()
        df["roll_min_cp"] = df["low"].rolling(10).min()
        df["roll_max_vol"] = df["volume"].rolling(10).max()




        # df.columns = [str(col) + (f'_{period}' if 'date' not in col else '') for col in df.columns]
        return df


class Trade:
    """ This class will trade the data from get_data class in interactive brokers. It includes strategy,
    buying/selling criteria, and controls all connections to interactive brokers orders.
    """

    def __init__(self):


        self.call_cost = -1
        self.put_cost = -1
        self.portfolio = []
        self.connect()
        self.ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        contract = Future(symbol='ES', lastTradeDateOrContractMonth='20210917', exchange='GLOBEX', currency='USD')  # define
        ib.qualifyContracts(contract)

        self.ES = ib.reqHistoricalData(contract=contract, endDateTime='', durationStr='2 D',
                                       barSizeSetting='3 mins', whatToShow='TRADES', useRTH=False, keepUpToDate=True,
                                       timeout=10)  # start data collection for ES-Mini
        df_raw = util.df(self.ES)


        df_1= df_raw.set_index('date')
        df_5 = df_1.resample('5T').agg(self.ohlc_dict)

        df_5.columns = ['open','high','low','close','volume']

        df_1 = res.tech_analysis(df_1,1)
        df_5 = res.tech_analysis(df_5, 5)
        self.data = pd.merge(df_1,df_5,on='date', how='outer').fillna(method='ffill')
        self.data_raw = self.data
        self.stock_owned = np.zeros(2)  # get data from get data class
        self.option_position()  # check holding positions and initiate contracts for calls and puts
        ib.sleep(1)
        self.call_option_volume = np.ones(20)  # start call options volume array to get the max volume in the last 20

        self.put_option_volume = np.ones(20)  # start put options volume array to get the max volume in the last 20 ticks

        self.submitted = 0  # order submission flag
        self.portfolio = ib.portfolio()

        self.put_contract_price = 0.25 * round(
            ((self.put_option_price.ask + self.put_option_price.bid) / 2) / 0.25)  # calculate average put price
        self.call_contract_price = 0.25 * round(
            ((self.call_option_price.ask + self.call_option_price.bid) / 2) / 0.25)  # calculate average call price
        self.options_price = np.array(
            [self.call_contract_price, self.put_contract_price])  # set an array for options prices

        self.max_call_price = self.call_option_price.bid  # define max call price (use to compare to current price)

        self.max_put_price = self.put_option_price.bid  # define max put price (use to compare to current price)

        self.prev_cash = 0
        self.cash_in_hand = 0
        self.total_liquidity = 0
        self.portfolio_value = 0
        self.unrealizedPNL = 0
        self.realizedPNL = 0
        self.cash_in_hand = 0
        self.realizedPNL = 0
        self.unrealizedPNL = 0
        self.portfolio_value = 0
        self.barnumb_lock = False
        self.barnumb_value = 0
        for self.account in ib.accountValues():  # get initial account value
            self.cash_in_hand = float(
                self.account.value) if (
                        self.account.tag == 'TotalCashValue' and self.account.account == 'DU1347520') else self.cash_in_hand
            self.portfolio_value = float(
                self.account.value) if (
                        self.account.tag == 'GrossPositionValue' and self.account.account == 'DU1347520') else self.portfolio_value
            self.unrealizedPNL = float(
                self.account.value) if (
                        self.account.tag == 'UnrealizedPnL' and self.account.account == 'DU1347520') else self.unrealizedPNL
            self.realizedPNL = float(
                self.account.value) if (
                        self.account.tag == 'RealizedPnL' and self.account.account == 'DU1347520') else self.realizedPNL
        self.reqId = []
        self.second_buy = False
        ib.reqGlobalCancel()  # Making sure all orders for buying selling are canceled before starting trading

    def trade(self, ES, hasNewBar=None):
        # if not hasNewBar:
        #     return
        if self.submitted == 1:
            print('working on an order, wait please')
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            return

        df_raw = util.df(self.ES)

        df_raw.set_index('date', inplace=True)

        df_raw = res.tech_analysis(df_raw, 1)

        self.data_raw = df_raw

        if self.data_raw.iloc[-1, 1] == 0:
            return

        df = self.data_raw[
            ['high', 'low', 'close', 'volume',
             'roll_max_cp',
             'roll_min_cp', 'roll_max_vol', 'atr', 'obv_slope', 'bar_num']].tail(
            20)  # filter data

        if self.stock_owned.any() > 0 and not np.isnan(self.max_call_price) and not np.isnan(
                self.max_put_price):
            self.max_call_price = self.call_option_price.bid if self.call_option_price.bid > self.max_call_price else \
                self.max_call_price
            self.max_put_price = self.put_option_price.bid if self.put_option_price.bid > self.max_put_price else \
                self.max_put_price  # check if holding positions and how much the max price for current position

        else:
            self.max_call_price = self.call_option_price.bid

            self.max_put_price = self.put_option_price.bid

        if self.stock_owned[0] > 0:
            print(f'Call cost was = {self.call_cost}')
            print((self.call_option_price.bid - self.call_cost))

        elif self.stock_owned[1] > 0:
            print(f'Put cost was = {self.put_cost}')
            print((self.put_option_price.bid - self.put_cost))

        buy_index, sell_index, take_profit = self.strategy(df)  # set initial buy index to None

        print(f'stocks owning = {self.stock_owned}')
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        if not len(sell_index) == 0:  # start selling to stop loss

            if len(buy_index) == 0:
                for i in sell_index:
                    # self.stock_owned[i] = 0

                    if len(self.portfolio) > 0:
                        contract = self.call_contract if i == 0 else self.put_contract
                        ib.qualifyContracts(contract)
                        price = ib.reqMktData(contract, '', False, False, None)
                        self.flatten_position(contract, price)
                self.submitted = 0


            else:
                for i in sell_index:
                    # self.stock_owned[i] = 0

                    if len(self.portfolio) > 0:
                        contract = self.call_contract if i == 0 else self.put_contract
                        ib.qualifyContracts(contract)
                        price = ib.reqMktData(contract, '', False, False, None)
                        self.flatten_position(contract, price)
                ib.sleep(0)
                for i in buy_index:
                    contract = res.get_contract('C', 2000) if i == 0 else res.get_contract('P', 2000)
                    ib.qualifyContracts(contract)

                    if self.cash_in_hand > (self.options_price[i] * 50) and self.cash_in_hand > self.portfolio_value \
                            and (self.stock_owned[0] < 1 or self.stock_owned[1] < 1) and len(
                        self.portfolio) == 0:
                        price = ib.reqMktData(contract, '', False, False)
                        ib.sleep(1)
                        quantity = int((self.cash_in_hand / (self.options_price[i] * 50))) - 1 if \
                            int((self.cash_in_hand / (self.options_price[i] * 50))) > 1 else 1
                        self.block_buying = 1
                        self.open_position(contract=contract, quantity=quantity, price=price)
                self.submitted = 0
            self.second_buy = False



        elif not len(take_profit) == 0:  # start selling to take profit
            for i in take_profit:

                print(self.stock_owned[i])
                print(len(self.portfolio))

                if len(self.portfolio) > 0:
                    contract = self.call_contract if i == 0 else self.put_contract
                    ib.qualifyContracts(contract)
                    price = ib.reqMktData(contract, '', False, False, None)
                    self.take_profit(contract, price)
            self.submitted = 0



        elif not len(buy_index) == 0:  # start buying to start trade
            if self.stock_owned.any() > 4:
                print('cancel buying too many contracts')
                return
            print(f'buying index = {buy_index}')
            for i in buy_index:
                if not self.stock_owned.any() > 0:
                    contract = res.get_contract('C', 2000) if i == 0 else res.get_contract('P', 2000)
                    ib.qualifyContracts(contract)
                else:

                    contract = self.call_contract if i == 0 else self.put_contract

                if self.cash_in_hand > (self.options_price[i] * 50) \
                        and (self.stock_owned[0] < 2 or self.stock_owned[1] < 2):
                    price = ib.reqMktData(contract, '', False, False)
                    ib.sleep(1)
                    quantity = int((self.cash_in_hand / (self.options_price[i] * 50))) - 1 if \
                        int((self.cash_in_hand / (self.options_price[i] * 50))) > 1 else ib.positions()[
                        0].position if self.second_buy is True else 1
                    self.block_buying = 1
                    self.open_position(contract=contract, quantity=quantity, price=price)
            self.submitted = 0

    def strategy(self, df):
        """
        Strategy to trade is:

            Opening positions if:
                - Buying ES Calls options when ES breaks the resistance from the last 30 minutes and the volume is
                higher than the last 30 minutes

                - Buying ES Puts options when ES breaks the support from the last 30 ninutes and the volume is
                higher than the last 30 minutes


            Closing positions if:

                - For calls:
                    * Candle's previous close price - candle's previous atr was higher than current candles' low price
                    * The current call option's price is less than 0.5 from the highest price and current candle's
                      OBV slope angle is less than 0
                - For puts:
                    * Candle's previous close price + candle's previous atr was lower than current candles' high price
                    * The current put option's price is less than 0.5 from the highest price and current candle's
                      OBV slope angle is more than 0

        """

        buy_index = []  # set initial buy index to None
        sell_index = []  # set initial sell index to None
        take_profit = []  # set initial take profit index to None
        i = -1  # use to get the last data in dataframe+
        print( f'volume this minute so far = {df["volume"].iloc[i]}, max volume last 10 minutes = {df["roll_max_vol"].iloc[i-1]}')
        print(f'price = {df["low"].iloc[i - 1]}, max_high ={df["roll_max_cp"].iloc[i-1]}, max_low = {df["roll_min_cp"].iloc[i - 1]}')
        print('buying put', df["low"].iloc[i] <= df["roll_min_cp"].iloc[i - 1] ,
                df["volume"].iloc[i] >df["roll_max_vol"].iloc[i - 1])
        print('buying call',(df['high'].iloc[i] >= df["roll_max_cp"].iloc[i-1]), \
                df["volume"].iloc[i]>df["roll_max_vol"].iloc[i-1])
        print(
            f'bar numb = {self.barnumb_lock} and self.barnumb_value= {self.barnumb_value} df["bar_num"] = {df["bar_num"].iloc[-1]}')
        print(f'max call price = {self.max_call_price} and max put price= {self.max_put_price} and obv slope = {df["obv_slope"].iloc[i]}')
        print(f'current call bid price = {self.call_option_price.bid} and current put bid price = {self.put_option_price.bid}')

        if (self.portfolio_value != 0 and self.stock_owned[0] == 0 and self.stock_owned[1] == 0) or (
                self.stock_owned[0] != 0 or self.stock_owned[1] != 0 and self.portfolio_value == 0):
            self.option_position()
            self.submitted = 0

        if self.call_option_price.bid < 1.25 or np.isnan(self.call_option_price.bid) or self.put_option_price.bid < 1.25 \
                or np.isnan(self.put_option_price.bid) or (self.data_raw.iloc[-1, 2] < 100):
            print('glitch or slippage in option prices, cancel check')
            return buy_index, sell_index, take_profit


        elif (self.stock_owned[0] == 0 and self.stock_owned[1] == 0) and (
                df['high'].iloc[i-1] >= df["roll_max_cp"].iloc[i-1] and
                df["volume"].iloc[i]>0.6*df["roll_max_vol"].iloc[i-1]) and buy_index == [] and self.submitted == 0:
            print("Buy call")
            buy_index.append(0)
            self.submitted = 1
            return buy_index, sell_index, take_profit

        elif (self.stock_owned[0] == 0 and self.stock_owned[1] == 0) and (
                df["low"].iloc[i-1]<= df["roll_min_cp"].iloc[i-1] and
                df["volume"].iloc[i]>0.6*df["roll_max_vol"].iloc[i-1]) and buy_index == [] and self.submitted == 0:
            print("Buy put")
            buy_index.append(1)
            self.submitted = 1
            return buy_index, sell_index, take_profit

        elif (self.stock_owned[0] >= 1) and not np.isnan(self.call_option_price.bid) and \
                ((df['low'].iloc[i]<df['close'].iloc[i-1] - df['atr'].iloc[i-1]) or (self.call_option_price.bid < self.max_call_price and df['obv_slope'].iloc[i] <= 0))\
                 and \
                self.call_option_price.bid > self.call_option_price.modelGreeks.optPrice and self.submitted == 0:

            # conditions to sell calls to stop loss
            self.submitted = 1
            print("sell call")
            sell_index.append(0)

            return buy_index, sell_index, take_profit

        elif (self.stock_owned[1] >= 1) and not np.isnan(self.put_option_price.bid) and \
                ((df["high"].iloc[i]>df['close'].iloc[i-1] + df['atr'].iloc[i-1]) or (self.put_option_price.bid < self.max_put_price and df['obv_slope'].iloc[i] >= 0))\
                 and \
                self.put_option_price.bid > self.put_option_price.modelGreeks.optPrice and self.submitted == 0:
            # conditions to sell puts to stop loss

            print("sell put")
            sell_index.append(1)
            self.submitted = 1
            return buy_index, sell_index, take_profit

        # elif (self.stock_owned[0] >= 1) and not np.isnan(self.call_option_price.bid) and \
        #         df['low'].iloc[i] <= df["roll_min_cp"].iloc[i - 1] and \
        #         df['volume'].iloc[i] >0.6*df["roll_max_vol"].iloc[i - 1] and \
        #         self.call_option_price.bid > self.call_option_price.modelGreeks.optPrice and self.submitted == 0:
        #
        #     self.submitted = 1
        #     print("sell call buy put")
        #     sell_index.append(0)
        #     buy_index.append(1)
        #
        #     return buy_index, sell_index, take_profit
        # #
        #
        # elif (self.stock_owned[1] >= 1) and not np.isnan(self.put_option_price.bid) and \
        #         df["high"].iloc[i] >= df["roll_max_cp"].iloc[i] and \
        #         df['volume'].iloc[i] >0.6*df["roll_max_vol"].iloc[i - 1] and \
        #         self.put_option_price.bid > self.put_option_price.modelGreeks.optPrice and self.submitted == 0:
        #     # conditions to sell puts to stop loss
        #
        #     print("sell put buy call")
        #     sell_index.append(1)
        #     buy_index.append(0)
        #     self.submitted = 1
        #     return buy_index, sell_index, take_profit
        elif self.barnumb_lock is True and self.barnumb_value != self.data_raw["bar_num"].iloc[i]:
            self.submitted = 0
            self.barnumb_lock = False
            self.barnumb_value = 0

            return buy_index, sell_index, take_profit
        else:
            print("Hold")
            return buy_index, sell_index, take_profit

    def error(self, reqId=None, errorCode=None, errorString=None, contract=None):  # error handler
        print(errorCode, errorString)

        if errorCode in [2104, 2108, 2158, 10182, 1102, 2106, 2107] and len(self.reqId) < 1:
            self.reqId.append(reqId)
            ib.cancelHistoricalData(self.ES)
            del self.ES
            ib.sleep(30)
            ES = Future(symbol='ES', lastTradeDateOrContractMonth='20210917', exchange='GLOBEX',
                        currency='USD')  # define
            # ES-Mini futures contract
            ib.qualifyContracts(ES)
            self.ES = ib.reqHistoricalData(contract=ES, endDateTime='', durationStr='2 D',
                                           barSizeSetting='3 mins', whatToShow='TRADES', useRTH=False, keepUpToDate=True,
                                           timeout=10)  # start data collection for ES-Mini
            print('attempt to restart data check')
            if len(self.ES) == 0:
                print(self.ES)
                self.error()
                self.reqId = []
            else:
                ib.sleep(1)
                self.reqId = []
                self.ES.updateEvent += self.trade
                self.trade(self.ES)

        elif errorCode == 201:
            self.option_position()

    def flatten_position(self, contract, price):  # flat position to stop loss

        print('flatttttttttttttttttttttttttttttttttttttttttttttttttttttt')
        portfolio = self.portfolio
        for each in portfolio:  # check current position and select contract
            print(price.bid)
            if each.contract != contract:
                if contract.right == 'C':
                    self.call_contract = each.contract
                elif contract.right == 'P':
                    self.put_contract = each.contract
                return
            ib.qualifyContracts(each.contract)

            action = 'SELL'  # to offset the long portfolio

            totalQuantity = abs(each.position)  # check holding quantity

            print(f'price = {price.bid + 0.25}')
            print(f'Flatten Position: {action} {totalQuantity} {contract.localSymbol}')
            order = LimitOrder(action=action, totalQuantity=totalQuantity, lmtPrice=x_round((price.ask + price.bid)/2),
                               account='U2809143') if each.position > 0 \
                else MarketOrder(action=action, totalQuantity=totalQuantity,
                                 account='U2809143')  # closing position as fast as possible
            trade = ib.placeOrder(each.contract, order)
            ib.sleep(10)  # waiting 10 secs
            if not trade.orderStatus.remaining == 0:
                ib.cancelOrder(order)  # canceling order if not filled
                self.submitted = 0
            else:
                if trade.orderStatus.status == 'Filled':
                    self.barnumb_lock = True
                    self.barnumb_value = self.data_raw['bar_num'].iloc[-1]
                self.submitted = 0

            print(trade.orderStatus.status)

        ib.sleep(0)

        return

    def take_profit(self, contract, price):  # start taking profit
        if np.isnan(price.bid) or self.stock_owned.any()==1:
            self.submitted = 0
            return
        print('take_________________profit')
        portfolio = self.portfolio
        for each in portfolio:
            if each.contract != contract:
                if contract.right == 'C':
                    self.call_contract = each.contract
                elif contract.right == 'P':
                    self.put_contract = each.contract
                return
            # if (price.bid - 0.5) <= 0.25 + (each.averageCost / 50):  # check if profit did happen
            #     print(price.bid, each.averageCost / 50)
            #     print('cancel sell no profit yet')
            #     self.submitted = 0
            #     return
            ib.qualifyContracts(each.contract)

            action = 'SELL'  # to offset the long portfolio

            totalQuantity = abs(each.position)

            print(f'price = {price.bid}')
            print(f'Take profit Position: {action} {totalQuantity} {contract.localSymbol}')

            order = LimitOrder(action=action, totalQuantity=totalQuantity, lmtPrice=x_round((price.ask + price.bid)/2), account='U2809143')
            trade = ib.placeOrder(each.contract, order)
            ib.sleep(15)
            if not trade.orderStatus.remaining == 0:
                ib.cancelOrder(order)
                self.submitted = 0
            else:
                self.barnumb_value = self.data_raw['bar_num'].iloc[-1]
                self.barnumb_lock = True
                self.submitted = 0
            print(trade.orderStatus.status)

        return

    def open_position(self, contract, quantity, price):  # start position
        import math

        if len(ib.positions()) > 0 or len(ib.reqAllOpenOrders()) > 0 :
            # if (len(ib.positions()) > 0 or len(ib.reqAllOpenOrders()) > 0) and (self.second_buy is False):
            print('Rejected to buy, either because the time of trade or there is another order or current loss >= 200')
            self.submitted = 0
            return
        quantity = 4 if int(math.floor(price.bid*50 / (float(self.cash_in_hand)))) > 4 else 1
        order = LimitOrder(action='BUY', totalQuantity=quantity,
                           lmtPrice=price.ask, account='U2809143')  # round(25 * round(price[i]/25, 2), 2))
        trade = ib.placeOrder(contract, order)
        print(f'buying {"CALL" if contract.right == "C" else "PUT"}')
        ib.sleep(15)
        if not trade.orderStatus.status == "Filled":
            ib.cancelOrder(order)
            self.submitted = 0
        else:
            self.stock_owned = np.array([quantity, 0]) if contract.right == "C" else np.array([0, quantity])
            self.second_buy = False
            self.submitted = 0

        self.submitted = 0
        print(trade.orderStatus.status)
        return

    def option_position(self, event=None):
        position = ib.portfolio()
        call_position = None
        put_position = None
        if len(position) == 0:
            self.stock_owned = np.zeros(2)
            self.portfolio = position
            self.call_cost = -1
            self.put_cost = -1

            self.call_contract = res.get_contract('C', 2000)
            ib.qualifyContracts(self.call_contract)

            self.put_contract = res.get_contract('P', 2000)
            ib.qualifyContracts(self.put_contract)

            self.call_option_price = ib.reqMktData(self.call_contract, '', False,
                                                   False)  # start data collection for calls
            self.put_option_price = ib.reqMktData(self.put_contract, '', False, False)  # start data collection for puts

            ib.sleep(1)

            return
        else:
            if self.call_cost or self.put_cost:
                pass
            if self.portfolio != position:
                self.portfolio = position

                for each in position:
                    if each.contract.right == 'C':
                        call_position = each.contract
                        put_position = None
                        ib.qualifyContracts(call_position)
                        self.stock_owned[0] = each.position
                        self.call_cost = 0.25 * round(each.averageCost / 50 / 0.25)
                    elif each.contract.right == 'P':
                        put_position = each.contract
                        call_position = None
                        ib.qualifyContracts(put_position)
                        self.stock_owned[1] = each.position
                        self.put_cost = 0.25 * round(each.averageCost / 50 / 0.25)

                self.call_cost = self.call_cost if not isinstance(call_position, type(None)) else -1
                self.put_cost = self.put_cost if not isinstance(put_position, type(None)) else -1

                self.call_contract = call_position if not isinstance(call_position, type(None)) else res.get_contract(
                    'C', 2000)
                ib.qualifyContracts(self.call_contract)

                self.put_contract = put_position if not isinstance(put_position, type(None)) else res.get_contract('P',
                                                                                                                   2000)
                ib.qualifyContracts(self.put_contract)

                self.call_option_price = ib.reqMktData(self.call_contract, '', False,
                                                       False)  # start data collection for calls
                self.put_option_price = ib.reqMktData(self.put_contract, '', False,
                                                      False)  # start data collection for puts
                ib.sleep(0)
                return
            else:
                self.portfolio = position
                return

    @staticmethod
    def connect():
        ib.disconnect()
        ib.connect('127.0.0.1', 7496, clientId=np.random.randint(10, 1000))
        ib.client.MaxRequests = 55
        print('reconnected')

    @staticmethod
    def roll_contract(option_vol, value):
        option_vol = np.roll(option_vol, -1)
        option_vol[-1] = value
        return option_vol

    def account_update(self, value=None):

        self.cash_in_hand = float(
            value.value) if value.tag == 'TotalCashValue' and value.account == 'DU1347520' else self.cash_in_hand

        self.portfolio_value = float(
            value.value) if value.tag == 'GrossPositionValue' and value.account == 'DU1347520' else self.portfolio_value
        self.unrealizedPNL = float(
            value.value) if value.tag == 'UnrealizedPnL' and value.account == 'DU1347520' else self.unrealizedPNL
        self.realizedPNL = float(
            value.value) if value.tag == 'RealizedPnL' and value.account == 'DU1347520' else self.realizedPNL

        if self.prev_cash != self.cash_in_hand:

            self.prev_cash = self.cash_in_hand
            if self.submitted == 1:
                self.submitted = 0


def is_time_between(begin_time, end_time, check_time=None):
    # If check time is not given, default to current UTC time
    check_time = check_time or datetime.now().time()
    if begin_time < end_time:
        return begin_time <= check_time <= end_time
    else:  # crosses midnight
        return check_time >= begin_time or check_time <= end_time


def main():
    ib.positionEvent += trading.option_position
    # ib.updatePortfolioEvent += trading.option_position
    ib.accountValueEvent += trading.account_update
    ib.errorEvent += trading.error
    trading.ES.updateEvent += trading.trade
    ib.run()


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    ib = IB()
    import os

    path = os.getcwd()
    TRADES_FOLDER = f'{path}/trades_logs'
    maybe_make_dir(TRADES_FOLDER)
    my_file = os.path.join(TRADES_FOLDER, f'/log_{datetime.strftime(datetime.now(), "%m_%d_%H_%M")}.txt')
    if not os.path.exists(my_file):
        file = open(f'{TRADES_FOLDER}/log_{datetime.strftime(datetime.now(), "%m_%d_%H_%M")}.txt', 'a+',
                    encoding='utf-8')
    # file = open(os.path.dirname(TRADES_FOLDER) + f'/log_{datetime.strftime(datetime.now(), "%m_%d_%H_%M")}.txt')
    while is_time_between(time(14, 00),
                          time(15, 00)):
        wait_time = 60 - datetime.now().minute
        print(f"wait until market opens in {wait_time} minutes")
        ib.sleep(60)

    res = get_data()
    trading = Trade()
    try:
        main()
    except ValueError:
        ib.sleep(5)
        main()
    except Exception as e:
        print(e)
        ib.disconnect()
        file.close()
    except "peer closed connection":
        ib.sleep(5)
        main()

    except "asyncio.exceptions.TimeoutError":
        ib.sleep(5)
        main()
    except KeyboardInterrupt:
        print('User stopped running')
        ib.disconnect()
        file.close()
