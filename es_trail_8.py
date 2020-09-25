import numpy as np
import pandas as pd
import sys
import asyncio
from datetime import datetime, timedelta
import talib as ta
from talib import MA_Type
from ib_insync import *
from ressup import ressup
import nest_asyncio
nest_asyncio.apply()
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

class Trade():
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

        # self.account_update()



    def flatten_position(self, contract, price):

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
            ib.sleep(5)
            if not trade.orderStatus.remaining == 0:
                ib.cancelOrder(order)
            print(trade.orderStatus.status)
            return

    def option_position(self, event=None):
        self.stock_owned = np.zeros(2)
        position = ib.portfolio()
        call_position = None
        put_position = None
        for each in position:
            if each.contract.right == 'C':
                call_position = each.contract
                ib.qualifyContracts(call_position)
                self.stock_owned[0] = each.position
            elif each.contract.right == 'P':
                put_position = each.contract
                ib.qualifyContracts(put_position)
                self.stock_owned[1] = each.position
        self.call_contract = call_position if not pd.isna(call_position) else res.get_contract('C', 2000)
        ib.qualifyContracts(self.call_contract)
        self.put_contract = put_position if not pd.isna(put_position) else res.get_contract('P', 2000)
        ib.qualifyContracts(self.put_contract)


    def trade(self, ES, hasNewBar=None):
        buy_index = []
        sell_index = []
        tickers_signal = "Hold"
        ib.sleep(0)
        self.option_position()
        cash_in_hand = float(ib.accountSummary()[22].value)
        portolio_value = float(ib.accountSummary()[29].value)

        call_contract_price = (self.call_option_price.ask + self.call_option_price.bid) / 2
        put_contract_price = (self.put_option_price.ask + self.put_option_price.bid) / 2
        options_array = np.array([call_contract_price, put_contract_price])
        self.call_option_volume = self.roll_contract(self.call_option_volume, self.call_option_price.bidSize)
        self.put_option_volume = self.roll_contract(self.put_option_volume, self.put_option_price.bidSize)
        # options_bid_volume = np.array([self.call_option_volume,self.put_option_volume])
        data_raw = res.ES(ES)

        df = data_raw[
            ['high', 'low', 'volume', 'close', 'RSI', 'ATR', 'roll_max_cp', 'roll_min_cp', 'roll_max_vol']].tail()

        print(
            f'cash in hand = {cash_in_hand}, portfolio value = {portolio_value}, unrealized PNL = {ib.accountSummary()[32].value}, realized PNL = {ib.accountSummary()[33].value}, holding = {self.stock_owned[0]} calls and {self.stock_owned[1]} puts and ES = {data_raw.iloc[-1, 3]} and [call,puts] values are = {options_array}')
        if df["high"].iloc[-1] >= df["roll_max_cp"].iloc[-2] and \
                df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and \
                len(ib.portfolio()) == 0 and buy_index == []:

            tickers_signal = "Buy call"
            buy_index.append(0)

        elif df["low"].iloc[-1] <= df["roll_min_cp"].iloc[-2] and \
                df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and \
                len(ib.portfolio()) == 0 and buy_index == []:

            tickers_signal = "Buy put"
            buy_index.append(1)

        elif df["low"].iloc[-1] <= df["roll_min_cp"].iloc[-2] and \
                df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and df['RSI'].iloc[-1] < 70 \
                and len(ib.portfolio()) != 0 and len(
            ib.reqAllOpenOrders()) == 0 and sell_index == [] and buy_index == []:
            tickers_signal = "sell call and buy puts"
            sell_index.append(0)
            buy_index.append(1)


        elif df["high"].iloc[-1] >= df["roll_max_cp"].iloc[-2] and \
                df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2] and df['RSI'].iloc[-1] > 30 \
                and len(ib.portfolio()) != 0 and len(
            ib.reqAllOpenOrders()) == 0 and sell_index == [] and buy_index == []:
            tickers_signal = "sell put and buy calls"
            sell_index.append(1)
            buy_index.append(0)

        elif (df["close"].iloc[-1] < df["close"].iloc[-2] - (1.5 * df["ATR"].iloc[-2]) \
              or (df["close"].iloc[-1] < df["low"].iloc[-2] and \
                  df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2]) or \
              2 <= self.call_option_volume[-1] <= self.call_option_volume.max() / 4) and \
                self.stock_owned[0] != 0 and len(ib.portfolio()) != 0 and len(ib.reqAllOpenOrders()) == 0 and \
                sell_index == [] and buy_index == []:
            print('1 ************************')
            tickers_signal = "sell call"
            sell_index.append(0)

        elif (df["close"].iloc[-1] > df["close"].iloc[-2] + (1.5 * df["ATR"].iloc[-2]) \
              or (df["close"].iloc[-1] > df["high"].iloc[-2] and \
                  df["volume"].iloc[-1] > df["roll_max_vol"].iloc[-2]) or \
              2 <= self.put_option_volume[-1] <= self.put_option_volume.max() / 4) and \
                self.stock_owned[1] != 0 and len(ib.portfolio()) != 0 and len(ib.reqAllOpenOrders()) == 0 and sell_index == [] and buy_index == []:
            print('2 ************************')
            tickers_signal = "sell put"
            sell_index.append(1)

        print(tickers_signal)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        if sell_index:

            for i in sell_index:
                print(self.stock_owned[i])

                if ((self.stock_owned[i] != 0 & i == 0) or (self.stock_owned[i] != 0 & i == 1)) and len(ib.reqAllOpenOrders()) == 0:

                    contract = self.call_contract if i == 0 else self.put_contract
                    ib.qualifyContracts(contract)
                    price = self.call_option_price if i == 0 else self.put_option_price
                    self.flatten_position(contract, price)

                    cash_in_hand = float(ib.accountSummary()[5].value)
                    sell_index = []

        if buy_index:

            for i in buy_index:
                contract = self.call_contract if i == 0 else self.put_contract
                ib.qualifyContracts(contract)

                if cash_in_hand > (options_array[i] * 50) and cash_in_hand > portolio_value \
                        and ((self.stock_owned[0] == 0 and i == 0) or (self.stock_owned[1] == 0 and i == 1)) and len(
                    ib.reqAllOpenOrders()) == 0 and len(ib.positions()) == 0:
                    options_array[i] = self.call_option_price.ask + 0.25 if i == 0 else self.put_option_price.ask + 0.25

                    quantity = 1  # int((cash_in_hand/(options_array[i] * 50)))

                    order = LimitOrder('BUY', quantity,
                                       options_array[i])  # round(25 * round(options_array[i]/25, 2), 2))
                    trade = ib.placeOrder(contract, order)
                    print(f'buying {"CALL" if contract.right == "C" else "PUT"}')
                    ib.sleep(5)
                    if not trade.orderStatus.remaining == 0:
                        ib.cancelOrder(order)
                    print(trade.orderStatus.status)

                    buy_index = []
                else:
                    buy_index = []

    def error(self, reqId, errorCode, errorString, contract):
        print(errorCode, errorString)
        if errorCode == 10197 or errorCode == 10182:
            for task in asyncio.Task.all_tasks():
                task.cancel()
            ib.disconnect()
            ib.sleep(3)


    def connect(self):
        ib.disconnect()
        ib.connect('127.0.0.1', 7497, clientId=np.random.randint(10, 1000))
        ib.client.MaxRequests = 55


    def roll_contract(self, option_vol, value):
        option_vol = np.roll(option_vol, -1)
        option_vol[-1] = value
        return option_vol

    # def account_update(self, value = None):
    #     self.account = ib.accountSummary()




def main():
    ib.positionEvent += trading.option_position
    # ib.accountValueEvent += trading.account_update
    ib.errorEvent += trading.error
    trading.ES.updateEvent += trading.trade
    ib.run()


if __name__ == '__main__':
    ib = IB()
    res = get_data()
    trading = Trade()
    while True:
        try:
            main()

        except:
            print('first try to connect failed waiting a minute and retry')
            try:
                trading.error()
                main()
            except:
                main()
