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

    @staticmethod
    def next_exp_weekday():
        """ Set next expiration date for contract 0 = Monday, 1 = Tuesday, etc..."""
        weekdays = {2: [5, 6, 0], 4: [0, 1, 2], 0: [3, 4]}
        today = datetime.today().weekday()
        for exp, day in weekdays.items():
            if today in day:
                return exp  # return the 2nd next weekday number

    @staticmethod
    def next_weekday(d, weekday):
        """ Translate weekdays number to a date for example next Mon = October 19th 2020"""
        days_ahead = weekday - d.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        date_to_return = d + timedelta(days_ahead)  # 0 = Monday, 1=Tus self.ES day, 2=Wed self.ES day...
        return date_to_return.strftime('%Y%m%d')  # return the date in the form of (yearmonthday) ex:(20201019)

    def get_strikes_and_expiration(self):
        """ When used, returns strikes and expiration for the ES futures options"""
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

    def ES(self, ES, ATR):
        """ By have the dataframe of ES futures, this function will analyze and
        provide technicals using TA-lib library"""

        ES_df = util.df(ES)
        ES_df.set_index('date', inplace=True)
        ES_df.index = pd.to_datetime(ES_df.index)
        # ES_df['hours'] = ES_df.index.strftime('%H').astype(int)
        # ES_df['minutes'] = ES_df.index.strftime('%M').astype(int)
        # ES_df['hours + minutes'] = ES_df['hours'] * 100 + ES_df['minutes']
        # ES_df['Day_of_week'] = ES_df.index.dayofweek
        ES_df['RSI'] = ta.RSI(ES_df['close'], timeperiod=5)
        # ES_df['macd'], ES_df['macdsignal'], ES_df['macdhist'] = ta.MACD(ES_df['close'], fastperiod=12, slowperiod=26,
        #                                                                signalperiod=9)
        # ES_df['macd - macdsignal'] = ES_df['macd'] - ES_df['macdsignal']
        # ES_df['MA_9'] = ta.MA(ES_df['close'], timeperiod=9)
        # ES_df['MA_21'] = ta.MA(ES_df['close'], timeperiod=21)
        # ES_df['MA_200'] = ta.MA(ES_df['close'], timeperiod=200)
        # ES_df['EMA_9'] = ta.EMA(ES_df['close'], timeperiod=9)
        # ES_df['EMA_26'] = ta.EMA(ES_df['close'], timeperiod=26)
        # ES_df['derv_1'] = np.gradient(ES_df['EMA_9'])
        # ES_df['EMA_9_26']=df['EMA_9']/df['EMA_26']
        # ES_df['EMA_50'] = ta.EMA(ES_df['close'], timeperiod=50)
        # ES_df['EMA_200'] = ta.EMA(ES_df['close'], timeperiod=200)
        ES_df['ATR'] = ta.ATR(ES_df['high'], ES_df['low'], ES_df['close'], timeperiod=20)
        ES_df['roll_max_cp'] = ES_df['high'].rolling(20).max()
        ES_df['roll_min_cp'] = ES_df['low'].rolling(20).min()
        # ES_df['Mean_ATR'] = (ta.ATR(ES_df['high'], ES_df['low'], ES_df['close'], 21)).mean()
        ES_df['roll_max_vol'] = ES_df['volume'].rolling(20).max()
        # ES_df['vol/max_vol'] = ES_df['volume'] / ES_df['roll_max_vol']
        ES_df['EMA_9-EMA_26'] = ES_df['EMA_9'] - ES_df['EMA_26']
        # ES_df['EMA_200-EMA_50'] = ES_df['EMA_200'] - ES_df['EMA_50']1
        ES_df['B_upper'], ES_df['B_middle'], ES_df['B_lower'] = ta.BBANDS(ES_df['close'], timeperiod=6, nbdevup=1,
                                                                          nbdevdn=1, matype=MA_Type.T3)
        ES_df.dropna(inplace=True)
        ES_df = renko_df(ES_df, ATR)
        return ES_df


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
        ES = Future(symbol='ES', lastTradeDateOrContractMonth='20201218', exchange='GLOBEX', currency='USD')  # define
        # ES-Mini futures contract
        ib.qualifyContracts(ES)
        self.ES = ib.reqHistoricalData(contract=ES, endDateTime='', durationStr='2 D',
                                       barSizeSetting='1 min', whatToShow='TRADES', useRTH=False, keepUpToDate=True,
                                       timeout=10)  # start data collection for ES-Mini
        df_raw = util.df(self.ES)
        self.ATR_data = df_raw
        self.ATR = (ta.ATR(df_raw['high'], df_raw['low'], df_raw['close'], 120)).mean()
        self.original_ATR = self.ATR
        self.data_raw = res.ES(self.ES, self.ATR)
        self.stock_owned = np.zeros(2)  # get data from get data class
        self.option_position()  # check holding positions and initiate contracts for calls and puts
        ib.sleep(1)
        self.call_option_volume = np.ones(20)  # start call options volume array to get the max volume in the last 20
        # self.call_option_price_average = np.ones(
        #     3)  # start call options volume array to get the max volume in the last 20
        self.put_option_volume = np.ones(
            20)  # start put options volume array to get the max volume in the last 20 ticks
        # self.put_option_price_average = np.ones(3)
        # self.call_option_price_average = self.roll_contract(self.call_option_price_average, self.call_option_price.bid)
        # self.put_option_price_average = self.roll_contract(self.put_option_price_average, self.put_option_price.bid)
        self.block_buying = 0  # Buying flag
        self.submitted = 0  # order submission flag
        self.portfolio = ib.portfolio()

        self.put_contract_price = 0.25 * round(
            ((self.put_option_price.ask + self.put_option_price.bid) / 2) / 0.25)  # calculate average put price
        self.call_contract_price = 0.25 * round(
            ((self.call_option_price.ask + self.call_option_price.bid) / 2) / 0.25)  # calculate average call price
        self.options_price = np.array(
            [self.call_contract_price, self.put_contract_price])  # set an array for options prices

        self.max_call_price = self.call_option_price.bid  # define max call price (use to compare to current price)
        # self.max_call_computation_price = self.call_option_price.modelGreeks.optPrice
        self.max_put_price = self.put_option_price.bid  # define max put price (use to compare to current price)
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
                self.account.value) if (self.account.tag == 'TotalCashValue' ) else self.cash_in_hand
            self.portfolio_value = float(
                self.account.value) if (self.account.tag == 'GrossPositionValue' ) else self.portfolio_value
            self.unrealizedPNL = float(
                self.account.value) if (self.account.tag == 'UnrealizedPnL' ) else self.unrealizedPNL
            self.realizedPNL = float(
                self.account.value) if (self.account.tag == 'RealizedPnL' ) else self.realizedPNL
        self.reqId = []
        self.ATR_factor = 1
        self.update = -1  # set this variable to -1 to get the last data in the get_data df
        self.ATR_minimum = self.ATR / 2
        self.ATR_decrement = 0.005
        self.barnumb_lock = False
        self.barnumb_value = 0
        self.second_calls_buy = False
        self.second_puts_buy = False
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

        self.call_option_volume = self.roll_contract(self.call_option_volume,
                                                     self.call_option_price.bidSize)  # update call options volume
        self.put_option_volume = self.roll_contract(self.put_option_volume,
                                                    self.put_option_price.bidSize)  # update put options volume
        # self.call_option_price_average = self.roll_contract(self.call_option_price_average, self.call_option_price.bid)
        # self.put_option_price_average = self.roll_contract(self.put_option_price_average, self.put_option_price.bid)

        self.data_raw = res.ES(self.ES, self.ATR)

        if self.data_raw.iloc[-1, 1] == 0:
            return
        df = self.data_raw[
            ['date', 'high', 'low', 'close', 'volume', 'bar_num', 'obv_slope', 'ATR', 'RSI', 'EMA_9-EMA_26',
             'roll_max_cp',
             'roll_min_cp', 'B_upper', 'B_middle', 'B_lower', 'roll_max_vol']].tail(
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
            print((self.call_option_price.bid / self.call_cost))

        elif self.stock_owned[1] > 0:
            print(f'Put cost was = {self.put_cost}')
            print((self.put_option_price.bid / self.put_cost))

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

            for i in buy_index:
                if not self.stock_owned.any() > 0:
                    contract = res.get_contract('C', 2000) if i == 0 else res.get_contract('P', 2000)
                    ib.qualifyContracts(contract)
                else:
                    contract = self.call_contract if i == 1 else self.put_contract

                if self.cash_in_hand > (self.options_price[i] * 50) \
                        and (self.stock_owned[0] < 2 or self.stock_owned[1] < 2):
                    price = ib.reqMktData(contract, '', False, False)
                    ib.sleep(1)
                    quantity = int((self.cash_in_hand / (self.options_price[i] * 50))) - 1 if \
                        int((self.cash_in_hand / (self.options_price[i] * 50))) > 1 else ib.positions()[
                        0].position if (self.second_calls_buy is True or self.second_puts_buy is True) else 1
                    self.block_buying = 1
                    self.open_position(contract=contract, quantity=quantity, price=price)
            self.submitted = 0



    def strategy(self, df):
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

        buy_index = []  # set initial buy index to None
        sell_index = []  # set initial sell index to None
        take_profit = []  # set initial take profit index to None
        i = -1  # use to get the last data in dataframe+

        # if self.stock_owned.any > 0 and datetime.now().minute // 15 == 0:
        #     self.discount = self.discount + 0.0025

        stop_loss = 2 + 1.75 + 0.25 * round((df["ATR"].iloc[i] ) / 0.25)  # set stop loss variable according to ATR
        self.ATR_factor = 0.25 * round((df["ATR"].iloc[i]) / 0.25) * 1.5

        print(
            f'time = {self.data_raw.iloc[-1, 0] - timedelta(hours=7)} cash in hand = {self.cash_in_hand}, portfolio value = {self.portfolio_value}, unrealized PNL ='
            f' {self.unrealizedPNL} realized PNL = {self.realizedPNL}, holding = {self.stock_owned[0]} '
            f'calls and {self.stock_owned[1]} puts and ES = {self.data_raw.iloc[-1, 1]} and bar_num = {df["bar_num"].iloc[-1]} and obv_slope = {df["obv_slope"].iloc[-1]}'
            f' and [call,puts] values are = '
            f'{self.options_price} and max call price = {self.max_call_price} compared to '
            f'{self.call_option_price.bid} and max put price = {self.max_put_price} compared to '
            f'{self.put_option_price.bid}'
            f'and ATR = {self.ATR} and ATR minimum = {self.ATR_minimum} and stop_loss = {stop_loss} and self.put_option_price.bid = '
            f'{self.put_option_price.bid} and EMA_9 - EMA_26 DIFF = {df["EMA_9-EMA_26"].iloc[i - 1]} and '
            f'RSI = {df["RSI"].iloc[i - 2]} and slop[-1] ={df["obv_slope"].iloc[-2]} and '
            f'self.submitted = {self.submitted} and upper BBand = {df["B_upper"].iloc[i - 1]} and'
            f' lower BBand = {df["B_lower"].iloc[i - 1]} and '
            f'df["roll_max_cp"] = {df["roll_max_cp"].iloc[i - 1]} and df["roll_min_cp"] = {df["roll_min_cp"].iloc[i - 1]}'
            f' and df["roll_max_vol"].iloc[i-1] = {df["roll_max_vol"].iloc[i - 1]} and '
            f'df["volume"].iloc[i-1] = {df["volume"].iloc[i - 1]}, and df["volume"] = {df["volume"].iloc[i]}'
            f' and barnum_lock = {self.barnumb_lock} and barnumb_bar = {self.barnumb_value}')

        if (self.portfolio_value != 0 and self.stock_owned[0] == 0 and self.stock_owned[1] == 0) or (
                self.stock_owned[0] != 0 or self.stock_owned[1] != 0 and self.portfolio_value == 0):
            self.option_position()
            self.submitted = 0

        if self.call_option_price.bid < 1.25 or np.isnan(self.call_option_price.bid) or self.put_option_price.bid < 1.25 \
                or np.isnan(self.put_option_price.bid) or (self.data_raw.iloc[-1, 1] < 100):
            print('glitch or slippage in option prices, cancel check')
            return buy_index, sell_index, take_profit

        elif self.barnumb_lock is False and self.stock_owned[0] == 0 and self.stock_owned[1] == 0 and \
                ((df["bar_num"].iloc[i - 1] > 2 and
                  df["obv_slope"].iloc[i - 1] > 25 and df['volume'].iloc[i] >= 0.5 * df['roll_max_vol'].iloc[
                      i - 1]) or (df["bar_num"].iloc[i] >= 2 and
                                  df["obv_slope"].iloc[i] > 38)) and \
                (df['B_upper'].iloc[i - 1] + 0.5 > df['close'].iloc[i - 1]) and \
                df['RSI'].iloc[-2] < 90 and df['EMA_9-EMA_26'].iloc[
            i - 1] > 0 and buy_index == [] and df['volume'].iloc[i] >= 0.5 * df['roll_max_vol'].iloc[
            i - 1] and self.submitted == 0:
            print("Buy call")
            buy_index.append(0)
            self.submitted = 1
            return buy_index, sell_index, take_profit

        elif self.barnumb_lock is False and self.stock_owned[0] == 0 and self.stock_owned[1] == 0 and \
                ((df["bar_num"].iloc[i - 1] < -2 and
                  df["obv_slope"].iloc[i - 1] < -25 and df['volume'].iloc[i] >= 0.5 * df['roll_max_vol'].iloc[
                      i - 1]) or (df["bar_num"].iloc[i] <= -2 and
                                  df["obv_slope"].iloc[i] < -38)) and \
                (df['B_lower'].iloc[i - 1] - 0.5 <= df['close'].iloc[i - 1]) and \
                df['RSI'].iloc[i - 2] > 10 and df['EMA_9-EMA_26'].iloc[
            i - 1] < 0 and buy_index == [] and self.submitted == 0:
            print("Buy put")
            buy_index.append(1)
            self.submitted = 1
            return buy_index, sell_index, take_profit

        elif self.stock_owned[0] == 0 and self.stock_owned[1] == 0 and \
                df["high"].iloc[i] >= df["roll_max_cp"].iloc[i - 1] and \
                df["volume"].iloc[i] > df["roll_max_vol"].iloc[i - 1] \
                and df["RSI"].iloc[i] < 80 and buy_index == [] and self.submitted == 0:
            #  conditions to buy calls
            print("Buy call")
            buy_index.append(0)
            self.submited = 1
            return buy_index, sell_index, take_profit

        elif self.stock_owned[0] == 0 and self.stock_owned[1] == 0 and \
                df["low"].iloc[i] <= df["roll_min_cp"].iloc[i - 1] and \
                df["volume"].iloc[i] > df["roll_max_vol"].iloc[i - 1] \
                and df["RSI"].iloc[i] > 20 and \
                buy_index == [] and self.submitted == 0:
            print("Buy put")
            buy_index.append(1)
            self.submitted = 1
            return buy_index, sell_index, take_profit

        elif (self.stock_owned[0] > 0 and self.stock_owned[1] == 0) \
                and (df["low"].iloc[i] <= df["roll_min_cp"].iloc[i - 1] and
                     df["volume"].iloc[i] > df["roll_max_vol"].iloc[i - 1]) and \
                self.stock_owned[1] == 0 and self.block_buying == 0 and self.submitted == 0:
            # conditions to sell calls and buy puts if trend reversed
            print("sell call and buy puts")
            sell_index.append(0)
            buy_index.append(1)
            self.submitted =1

            return buy_index, sell_index, take_profit

        elif (self.stock_owned[1] > 0 and self.stock_owned[0] == 0) and \
                (df["high"].iloc[i] >= df["roll_max_cp"].iloc[i - 1] and
                 df["volume"].iloc[i] > df["roll_max_vol"].iloc[i - 1]) \
                and self.stock_owned[0] == 0 and self.block_buying == 0 and self.submitted == 0:
            # conditions to buy calls and sell puts if trend reversed
            print("sell put and buy calls")
            sell_index.append(1)
            buy_index.append(0)
            self.submitted = 1
            return buy_index, sell_index, take_profit

        # elif (2 > self.stock_owned[0] > 0) and ((not np.isnan(self.call_option_price.bid)) and (
        #         ((self.call_option_price.bid / self.call_cost) <= 0.7) and not
        #         (df["bar_num"].iloc[i - 1] >= 2 and df["obv_slope"].iloc[i - 1] > 25) and
        #         self.call_option_price.bid > self.call_option_price.modelGreeks.optPrice)) and self.submitted == 0:
        #
        #     # conditions to sell calls to stop loss
        #     self.submitted = 1
        #     print("2nd buy calls")
        #     buy_index.append(0)
        #     self.second_calls_buy = True
        #
        #     return buy_index, sell_index, take_profit
        #
        # elif (2 > self.stock_owned[1] > 0) and ((not np.isnan(self.put_option_price.bid)) and (
        #         ((self.put_option_price.bid / self.put_cost) <= 0.7) and
        #         not (df["bar_num"].iloc[i - 1] <= -2 and df["obv_slope"].iloc[i - 1] < -25) and
        #         self.put_option_price.bid > self.put_option_price.modelGreeks.optPrice)) \
        #         and self.submitted == 0:
        #     # conditions to sell puts to stop loss
        #
        #     self.submitted = 1
        #     print("2nd buy puts")
        #     buy_index.append(1)
        #     self.second_puts_buy = True
        #     return buy_index, sell_index, take_profit

        elif self.stock_owned[0] > 0 and ((not np.isnan(self.call_option_price.bid)) and (
                ((self.call_option_price.bid / self.call_cost) < 0.9)) and not
                                            (df["bar_num"].iloc[i - 1] >= 2 and df["obv_slope"].iloc[i - 1] > 25) and
                                            self.call_option_price.bid > self.call_option_price.modelGreeks.optPrice) and self.submitted == 0:

            # conditions to sell calls to stop loss
            self.submitted = 1
            print("sell call")
            sell_index.append(0)

            return buy_index, sell_index, take_profit

        elif self.stock_owned[1] > 0 and ((not np.isnan(self.put_option_price.bid)) and (
                ((self.put_option_price.bid / self.put_cost) < 0.9)) and not
                                            (df["bar_num"].iloc[i - 1] <= -2 and df["obv_slope"].iloc[i - 1] < -25) and
                                            self.put_option_price.bid > self.put_option_price.modelGreeks.optPrice) and self.submitted == 0:
            # conditions to sell puts to stop loss

            print("sell put")
            sell_index.append(1)
            self.submitted = 1
            return buy_index, sell_index, take_profit

        elif (self.stock_owned[0] > 0) and (self.call_option_price.bid - self.call_cost >= 1) and (
                df["obv_slope"].iloc[i] < 30 or 2<self.call_option_volume[-1]<self.call_option_volume.max()/4) \
                and self.submitted == 0:

            print("take profits call")
            self.submitted = 1
            take_profit.append(0)
            return buy_index, sell_index, take_profit


        elif (self.stock_owned[1] > 0) and (self.put_option_price.bid - self.put_cost >= 1) and (
                df["obv_slope"].iloc[i] > -30 or 2<self.put_option_volume[-1]<self.put_option_volume.max()/4) \
                and self.submitted == 0:

            print("take profits puts")
            self.submitted = 1
            take_profit.append(1)
            return buy_index, sell_index, take_profit

        elif self.barnumb_lock is True and self.barnumb_value != df["bar_num"].iloc[i]:
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
            try:
                if is_time_between(time(17, 00),
                                   time(17, 15)) or each.contract.right != contract.right or price.bid < 1.25 or len(
                    ib.reqAllOpenOrders()) > 0 or not isinstance(price, Ticker) or (
                        np.isnan(price.modelGreeks.optPrice)) or \
                        price.modelGreeks.optPrice > price.bid:
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
                else MarketOrder(action=action, totalQuantity=totalQuantity)  # closing position as fast as possible
            trade = ib.placeOrder(each.contract, order)
            ib.sleep(10)  # waiting 10 secs
            if not trade.orderStatus.remaining == 0:
                ib.cancelOrder(order)  # canceling order if not filled
                self.submitted = 0
            else:
                self.submitted = 0
                self.ATR = self.original_ATR

            print(trade.orderStatus.status)

        ib.sleep(0)

        return

    def take_profit(self, contract, price):  # start taking profit
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

            order = LimitOrder(action=action, totalQuantity=totalQuantity, lmtPrice=price.bid)
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
        if len(ib.portfolio()) > 0 or len(ib.reqAllOpenOrders()) > 0:
            print('Rejected to buy, either because the time of trade or there is another order or current loss >= 200')
            self.submitted = 0
            return
        contract = self.call_contract if self.second_calls_buy is True else self.put_contract if self.second_puts_buy is True else contract
        quantity = 1  if (self.second_calls_buy is False or self.second_puts_buy is False) else 2 # quantity if quantity < 4 else 3
        order = LimitOrder(action='BUY', totalQuantity=quantity,
                           lmtPrice=price.ask)  # round(25 * round(price[i]/25, 2), 2))
        print(order)
        trade = ib.placeOrder(contract, order)
        print(trade)
        print(f'buying {"CALL" if contract.right == "C" else "PUT"}')
        ib.sleep(15)
        if not trade.orderStatus.status == "Filled":
            ib.cancelOrder(order)
            self.submitted = 0
        else:
            self.stock_owned = np.array([quantity, 0]) if contract.right == "C" else np.array([0, quantity])
            self.second_calls_buy = False
            self.second_puts_buy = False
            self.submitted = 0
            self.option_position()

        self.submitted = 0
        print(trade.orderStatus.status)
        self.ATR = self.original_ATR
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
        self.cash_in_hand = float(value.value) if value.tag == 'TotalCashValue' else self.cash_in_hand

        self.portfolio_value = float(value.value) if value.tag == 'GrossPositionValue' else  self.portfolio_value
        self.unrealizedPNL = float(value.value) if value.tag == 'UnrealizedPnL' else self.unrealizedPNL
        self.realizedPNL = float(value.value) if value.tag == 'RealizedPnL' else self.realizedPNL
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
