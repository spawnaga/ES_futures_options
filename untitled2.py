import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta, time, timezone
import talib as ta
from talib import MA_Type
from stocktrends import Renko
from ib_insync import *
import statsmodels.api as sm
import nest_asyncio

nest_asyncio.apply()  # enable nest asyncio
sys.setrecursionlimit(10 ** 9)  # set recursion limit to 1000000000
pd.options.mode.chained_assignment = None  # remove a warning


def slope(ser, n):
    "function to calculate the slope of n consecutive points on a plot"
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


def renko_df(df_raw, ATR):
    df_raw = df_raw[-500:]
    df_raw.reset_index(inplace=True)

    renko = Renko(df_raw[['date', 'open', 'high', 'low', 'close', 'volume']])

    renko.brick_size = ATR

    df = renko.get_ohlc_data()
    df['bar_num'] = np.where(df['uptrend'] == True, 1, np.where(df['uptrend'] == False, -1, 0))

    for i in range(1, len(df["bar_num"])):
        if df["bar_num"][i] > 0 and df["bar_num"][i - 1] > 0:
            df["bar_num"][i] += df["bar_num"][i - 1]
        elif df["bar_num"][i] < 0 and df["bar_num"][i - 1] < 0:
            df["bar_num"][i] += df["bar_num"][i - 1]
    df.drop_duplicates(subset="date", keep="last", inplace=True)
    df_raw = df_raw.merge(df.loc[:, ["date", "bar_num"]], how="outer", on="date")
    df_raw["bar_num"].fillna(method='ffill', inplace=True)
    df_raw['OBV'] = ta.OBV(df_raw['close'], df_raw['volume'])
    df_raw["obv_slope"] = slope(df_raw['OBV'], 5)
    return df_raw


class get_data:
    """ A class to get ES Technical analysis and next 2 days expiration date and delta 60 option strikes for
    whatever ES price at """

    def __init__(self):
        global trading
        self.trading = Trade
        self.trading.ATR_factor = 1

    def ES(self, dataframe, ATR):
        """ By have the dataframe of ES futures, this function will analyze and
        provide technicals using TA-lib library"""

        df = util.df(dataframe)
        df.set_index('date', inplace=True)
        dataframe['top'] = pd.Series(dataframe["close"]).rolling(125).quantile(0.25)
        dataframe['bot'] = pd.Series(dataframe["close"]).rolling(125).quantile(0.05)
        dataframe['adx'] = ta.ADX(dataframe,14)
        dataframe["roll_max_cp"] = dataframe["high"].rolling(20).max()
        dataframe["roll_min_cp"] = dataframe["low"].rolling(20).min()
        dataframe["roll_max_vol"] = dataframe["volume"].rolling(20).max()
        dataframe["adx_slope"] = slope(dataframe['adx'], 5)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=5)
        return df


# self = Trade
class Trade:
    """ This class will trade the data from get_data class in interactive brokers. It includes strategy,
    buying/selling criteria, and controls all connections to interactive brokers orders.
    """

    def __init__(self):

        self.skip = False
        self.call_cost = -1
        self.put_cost = -1
        self.portfolio = []
        self.connect()
        ES = Future(symbol='ES', lastTradeDateOrContractMonth='20210618', exchange='GLOBEX', currency='USD')  # define
        self.contract = ES
        # ES-Mini futures contract
        ib.qualifyContracts(ES)
        self.ES = ib.reqHistoricalData(contract=ES, endDateTime='', durationStr='2 D',
                                       barSizeSetting='1 min', whatToShow='TRADES', useRTH=False, keepUpToDate=True,
                                       timeout=10)  # start data collection for ES-Mini
        df_raw = util.df(self.ES)
        
        self.data_raw = res.ES(self.ES, ta.ATR(self.ES,14).mean())
        self.stock_owned = np.zeros(1)  # get data from get data class

        # self.put_option_price_average = np.ones(3)
        # self.call_option_price_average = self.roll_contract(self.call_option_price_average, self.call_option_price.bid)
        # self.put_option_price_average = self.roll_contract(self.put_option_price_average, self.put_option_price.bid)
        self.block_buying = 0  # Buying flag
        self.submitted = 0  # order submission flag
        self.portfolio = ib.portfolio()
        self.options_price = np.array(self.ES['close'].iloc[-1]*50)  # set an array for options prices

        # self.max_put_computation_price = self.put_option_price.modelGreeks.optPrice
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
        for self.account in ib.accountValues():  # get initial account value
            self.cash_in_hand = float(
                self.account.value) if (self.account.tag == 'BuyingPower' and self.account.account == 'U2809143') else self.cash_in_hand
            self.portfolio_value = float(
                self.account.value) if (self.account.tag == 'GrossPositionValue' and self.account.account == 'U2809143') else self.portfolio_value
            self.unrealizedPNL = float(
                self.account.value) if (self.account.tag == 'UnrealizedPnL' and self.account.account == 'U2809143') else self.unrealizedPNL
            self.realizedPNL = float(
                self.account.value) if (self.account.tag == 'RealizedPnL' and self.account.account == 'U2809143') else self.realizedPNL
        self.reqId = []
        self.ATR_factor = 1
        self.update = -1  # set this variable to -1 to get the last data in the get_data df
        self.ATR_minimum = self.ATR / 2
        self.ATR_decrement = 0.005
        self.barnumb_lock = False
        self.barnumb_value = 0
        self.second_buy = False
        # if self.stock_owned.any() > 0:
        #     self.discount =

        # self.call_option_price_average = self.roll_contract(self.call_option_price_average, self.call_option_price.bid)
        # self.put_option_price_average = self.roll_contract(self.put_option_price_average, self.put_option_price.bid)
        ib.reqGlobalCancel()  # Making sure all orders for buying selling are canceled before starting trading

    def trade(self, ES, hasNewBar=None):
        if self.submitted == 1:
            print('working on an order, wait please')
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            return

        # if self.stock_owned.any() > 0 and self.ATR > self.ATR_minimum:
        #     self.ATR -= self.ATR_decrement
        self.ES = ES

        self.data_raw = res.ES(self.ES, ta.ATR(self.ES,14).mean())

        if self.data_raw.iloc[-1, 1] == 0:
            return
        df = self.data_raw.tail(20) 

        long_index, short_index, close_index = self.strategy(df)  # set initial buy index to None

        print(f'stocks owning = {self.stock_owned}')
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        if not len(close_index) == 0:  # start selling to stop loss

            if len(long_index) == 0:
                for i in close_index:
                    # self.stock_owned[i] = 0

                    if len(self.portfolio) > 0:
                        contract = self.ES
                        ib.qualifyContracts(contract)
                        price = ib.reqMktData(contract, '', False, False, None)
                        self.flatten_position(contract, price)
                self.submitted = 0


            else:
                for i in close_index:
                    # self.stock_owned[i] = 0

                    if len(self.portfolio) > 0:
                        contract = self.call_contract if i == 0 else self.put_contract
                        ib.qualifyContracts(contract)
                        price = ib.reqMktData(contract, '', False, False, None)
                        self.flatten_position(contract, price)
                ib.sleep(0)
                for i in long_index:
                    contract = Future(symbol='ES', lastTradeDateOrContractMonth='20210618', exchange='GLOBEX', currency='USD')
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



        elif not len(close_index) == 0:  # start selling to take profit
            for i in close_index:

                print(self.stock_owned)
                print(len(self.portfolio))

                if len(self.portfolio) > 0:
                    contract = self.ES
                    ib.qualifyContracts(contract)
                    price = ib.reqMktData(contract, '', False, False, None)
                    self.close_index(contract, price)
            self.submitted = 0



        elif not len(long_index) == 0:  # start buying to start trade

            for i in long_index:
                if not self.stock_owned.any() == 0:
                    contract = self.ES
                    ib.qualifyContracts(contract)
                    price = ib.reqMktData(contract, '', False, False, None)


                if self.cash_in_hand > (price * 50) \
                        and (self.stock_owned[0] < 2 or self.stock_owned[1] < 2):
                    price = ib.reqMktData(contract, '', False, False)
                    ib.sleep(1)
                    quantity = int((self.cash_in_hand / (self.options_price[i] * 50))) - 1 if \
                        int((self.cash_in_hand / (self.options_price[i] * 50))) > 1 else ib.positions()[
                        0].position if self.second_buy is True else 1
                    self.block_buying = 1
                    self.open_position(contract=contract, quantity=quantity, price=price)
            self.submitted = 0



    def strategy(self, dataframe):
        """
        Strategy to trade is:

            Opening positions if:
                - Buying ES Calls options when ES breaks the resistance in the last 20 ticks
                - Buying ES Calls options if ES made 2 positive renkos bars while RSI is less than 90, EMA 9 > EMA 26,
                volume is more than 1/2 Max Volume of the last 20 candles and, ES price is less than the upper 1st Standard
                deviation of Bolinger Bands.

                - Buying ES Puts options when ES breaks the support in the last 20 ticks
                - Buying ES puts options if ES made 2 negative renkos bars while RSI is more than 20, EMA 9 < EMA 26,
                volume is more than 1/2 Max Volume of the last 20 candles and, ES price is more than the lower 1st Standard
                deviation of Bolinger Bands.


            Closing positions if:

                - ES trend reverses
                - ES's ATR is more than 1.25 in the opposite direction
                - Option price is less losing more than 1.25 * 0.5 * ATR from the cost price
                - Option price made 10% profits and obv volum indicators reverses to the other side

        """

        long_index = []  # set initial buy index to None
        short_index = []  # set initial sell index to None
        close_index = []  # set initial take profit index to None
        i = -1  # use to get the last data in dataframe+

        # if self.stock_owned.any > 0 and datetime.now().minute // 15 == 0:
        #     self.discount = self.discount + 0.0025

        stop_loss = 2 + 1.75 + 0.25 * round((df["ATR"].iloc[i] ) / 0.25)  # set stop loss variable according to ATR
        self.ATR_factor = 0.25 * round((df["ATR"].iloc[i]) / 0.25) * 1.5

        # print(
        #     f'time = {self.data_raw.iloc[-1, 0] - timedelta(hours=7)} cash in hand = {self.cash_in_hand}, portfolio value = {self.portfolio_value}, unrealized PNL ='
        #     f' {self.unrealizedPNL} realized PNL = {self.realizedPNL}, holding = {self.stock_owned[0]} '
        #     f'calls and {self.stock_owned[1]} puts and ES = {self.data_raw.iloc[-1, 1]} and bar_num = {df["bar_num"].iloc[-1]} and obv_slope = {df["obv_slope"].iloc[-1]}'
        #     f' and [call,puts] values are = '
        #     f'{self.options_price} and max call price = {self.max_call_price} compared to '
        #     f'{self.call_option_price.bid} and max put price = {self.max_put_price} compared to '
        #     f'{self.put_option_price.bid}'
        #     f'and ATR = {self.ATR} and ATR minimum = {self.ATR_minimum} and stop_loss = {stop_loss} and self.put_option_price.bid = '
        #     f'{self.put_option_price.bid} and EMA_9 - EMA_26 DIFF = {df["EMA_9-EMA_26"].iloc[i - 1]} and '
        #     f'RSI = {df["RSI"].iloc[i - 2]} and slop[-1] ={df["obv_slope"].iloc[-2]} and '
        #     f'self.submitted = {self.submitted} and upper BBand = {df["B_upper"].iloc[i - 1]} and'
        #     f' lower BBand = {df["B_lower"].iloc[i - 1]} and '
        #     f'df["roll_max_cp"] = {df["roll_max_cp"].iloc[i - 1]} and df["roll_min_cp"] = {df["roll_min_cp"].iloc[i - 1]}'
        #     f' and df["roll_max_vol"].iloc[i-1] = {df["roll_max_vol"].iloc[i - 1]} and '
        #     f'df["volume"].iloc[i-1] = {df["volume"].iloc[i - 1]}, and df["volume"] = {df["volume"].iloc[i]}'
        #     f' and barnum_lock = {self.barnumb_lock} and barnumb_bar = {self.barnumb_value}')

        if (self.portfolio_value != 0 and self.stock_owned == 0) or (
                self.stock_owned != 0 and self.portfolio_value == 0):
            self.contract_position()
            self.submitted = 0

        if self.contract_price.bid < 1.25 or np.isnan(self.contract_price.bid) or (self.data_raw.iloc[-1, 1] < 100):
            print('glitch or slippage in option prices, cancel check')
            return long_index_short_index_close_index

        elif self.stock_owned == 0 and dataframe['high'][i] >= 0.75*dataframe["roll_max_cp"][i] and \
            dataframe["volume"][i]>dataframe["roll_max_vol"][i-1] and long_index == [] and self.submitted == 0:
            print("Buy Long")
            long_index.append(0)
            self.submitted = 1
            return long_index_short_index_close_index

        elif self.stock_owned == 0 and  dataframe["low"][i]<=0.75*dataframe["roll_min_cp"][i] and \
            dataframe["volume"][i]>1.2*dataframe["roll_max_vol"][i-1] and long_index == [] and self.submitted == 0:
            print("Sell Short")
            short_index.append(0)
            self.submitted = 1
            return long_index_short_index_close_index


        elif (self.stock_owned > 0) and (dataframe['low'][i]<=0.75*dataframe["roll_min_cp"][i] and \
               dataframe['volume'][i]>1.2*dataframe["roll_max_vol"][i-1]) and self.block_buying == 0 and self.submitted == 0:
            # conditions to sell calls and buy puts if trend reversed
            print("Close Long and Start Short")
            close_index.append(0)
            short_index.append(0)
            self.submitted =1

            return long_index_short_index_close_index

        elif (self.stock_owned < 0) and(dataframe["high"][i]>=0.75*dataframe["roll_max_cp"][i] and \
           dataframe['volume'][i]>dataframe["roll_max_vol"][i-1]) and self.block_buying == 0 and self.submitted == 0:
            # conditions to buy calls and sell puts if trend reversed
            print("Close Short and Start Long")
            close_index.append(1)
            long_index.append(0)
            self.submitted = 1
            return long_index_short_index_close_index



        elif (self.stock_owned > 0) and not np.isnan(self.contract_price.bid) and (dataframe['low'][i]<dataframe['close'][i-1] - dataframe['atr'][i-1] or dataframe['obv_slope'][i] <= 25 or dataframe['bar_num'][i] <0) and self.submitted == 0:

            # conditions to sell calls to stop loss
            self.submitted = 1
            print("Close Long")
            close_index.append(0)

            return long_index_short_index_close_index

        elif (self.stock_owned < 0) and not np.isnan(self.contract_price.ask) and (dataframe["high"][i]>dataframe['close'][i-1] + dataframe['atr'][i-1] or dataframe['obv_slope'][i]>= -25  or dataframe['bar_num'][i] <0) and self.submitted == 0:
            # conditions to sell puts to stop loss

            print("Close Short")
            close_index.append(1)
            self.submitted = 1
            return long_index_short_index_close_index

        elif self.barnumb_lock is True and self.barnumb_value != df["bar_num"].iloc[i]:
            self.submitted = 0
            self.barnumb_value = False
            self.barnumb_value = 0
            return long_index_short_index_close_index

        else:
            print("Hold")
            return long_index_short_index_close_index

    def error(self, reqId=None, errorCode=None, errorString=None, contract=None):  # error handler
        print(errorCode, errorString)

        if errorCode in [2104, 2108, 2158, 10182, 1102, 2106, 2107] and len(self.reqId) < 1:
            self.reqId.append(reqId)
            ib.cancelHistoricalData(self.ES)
            del self.ES
            ib.sleep(50)
            ES = Future(symbol='ES', lastTradeDateOrContractMonth='20201218', exchange='GLOBEX',
                        currency='USD')  # define
            # ES-Mini futures contract
            ib.qualifyContracts(ES)
            self.ES = ib.reqHistoricalData(contract=ES, endDateTime='', durationStr='2 D',
                                           barSizeSetting='1 min', whatToShow='TRADES', useRTH=False, keepUpToDate=True,
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
            self.contract_position()

    def flatten_position(self, contract, price):  # flat position to stop loss

        print('flatttttttttttttttttttttttttttttttttttttttttttttttttttttt')
        portfolio = self.portfolio
        for each in portfolio:  # check current position and select contract
            print(price.bid)

            try:
                if is_time_between(time(17, 00),
                                   time(17, 15)) or each.contract.right != contract.right or price.bid < 1.25 or len(
                    ib.reqAllOpenOrders()) > 0:
                    print(f'Order to sell was rejected because of one of the following reasons:'
                          f'1- time Now is between 17:00 to 17:15 '
                          f'2- contract is not right'
                          f'3- contract price slippage (contract price dropped under its normal/real value)')
                    return
            except:
                print('checking price for slippage got an error, skipping sell to avoid loss, please check this part')
                return
            ib.qualifyContracts(each.contract)

            action = 'SELL'  # to offset the long portfolio

            totalQuantity = abs(each.position)  # check holding quantity

            print(f'price = {price.bid + 0.25}')
            print(f'Flatten Position: {action} {totalQuantity} {contract.localSymbol}')
            order = LimitOrder(action=action, totalQuantity=totalQuantity, lmtPrice=price.bid) if each.position > 0 \
                else LimitOrder(action=action, totalQuantity=totalQuantity, lmtPrice=price.ask)  # closing position as fast as possible
            trade = ib.placeOrder(each.contract, order)
            
            if not trade.orderStatus.remaining == 0:
                ib.cancelOrder(order)  # canceling order if not filled
                self.submitted = 0
            else:
                self.submitted = 0
                self.ATR = self.original_ATR

            print(trade.orderStatus.status)

        ib.sleep(0)

        return

    def close_index(self, contract, price):  # start taking profit
        if np.isnan(price.bid):
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
            if (price.bid - 0.5) <= 0.25 + (each.averageCost / 50):  # check if profit did happen
                print(price.bid, each.averageCost / 50)
                print('cancel sell no profit yet')
                self.submitted = 0
                return
            ib.qualifyContracts(each.contract)

            action = 'SELL'  # to offset the long portfolio

            totalQuantity = abs(each.position)

            print(f'price = {price.bid}')
            print(f'Take profit Position: {action} {totalQuantity} {contract.localSymbol}')

            order = LimitOrder(action=action, totalQuantity=totalQuantity, lmtPrice=price.bid, account='U2809143')
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
        if len(ib.positions()) > 0 or len(ib.reqAllOpenOrders()) > 0:
        # if (len(ib.positions()) > 0 or len(ib.reqAllOpenOrders()) > 0) and (self.second_buy is False):
            print('Rejected to buy, either because the time of trade or there is another order or current loss >= 200')
            self.submitted = 0
            return
        quantity = 1  # quantity if quantity < 4 else 3
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
        self.ATR = self.original_ATR
        return

    def contract_position(self, event=None):
        position = ib.portfolio()
        contract_position = None
        if len(position) == 0:
            self.stock_owned = np.zeros(1)
            self.portfolio = position
            self.contract_cost = -1

            self.contract = Future(symbol='ES', lastTradeDateOrContractMonth='20210618', exchange='GLOBEX', currency='USD')
            ib.qualifyContracts(self.contract)

            self.contract_price = ib.reqMktData(self.contract, '', False,
                                                   False)  # start data collection for calls
            ib.sleep(1)

            return
        

    @staticmethod
    def connect():
        ib.disconnect()
        ib.connect('127.0.0.1', 7497, clientId=np.random.randint(10, 1000))
        ib.client.MaxRequests = 55
        print('reconnected')

    @staticmethod
    def roll_contract(option_vol, value):
        option_vol = np.roll(option_vol, -1)
        option_vol[-1] = value
        return option_vol

    def account_update(self, value=None):

        self.update += 1
        self.cash_in_hand = float(value.value) if value.tag == 'TotalCashValue' and value.account=='U2809143' else self.cash_in_hand

        self.portfolio_value = float(value.value) if value.tag == 'GrossPositionValue' and value.account=='U2809143' else  self.portfolio_value
        self.unrealizedPNL = float(value.value) if value.tag == 'UnrealizedPnL' and value.account=='U2809143' else self.unrealizedPNL
        self.realizedPNL = float(value.value) if value.tag == 'RealizedPnL' and value.account=='U2809143' else self.realizedPNL
        # if self.update % 5 != 0:
        #     return
        # self.account = ib.accountSummary()

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
    ib.positionEvent += trading.contract_position
    # ib.updatePortfolioEvent += trading.contract_position
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
    while is_time_between(time(16, 00),
                          time(17, 00)):
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
