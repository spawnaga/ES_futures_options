
"""
Created on Sun Sep 20 18:27:11 2020

@author: alial
"""

import numpy as np
import pandas as pd

import sys
from datetime import datetime, timedelta
from ressup import ressup
from ib_insync import *

import nest_asyncio
nest_asyncio.apply()
# util.startLoop()
sys.setrecursionlimit(10**9)


class all_events:
    def account_update(self, value):
        print(value)
        return account

    def open_orders(self, trade):
        print(len(trade))
        return 
    
    def connected(self):
        print('is connected')
        
    def disconnected(self):
        print('is disconnected')
        
    def canceled_order(self, trade):
        print(trade.orderStatus.status)
        
    def portfolio_update(self, PortfolioItem):
        print(PortfolioItem)
        
    def position_update(self, position):
        print(position)
    
    def order_status(self, trade):
        print(trade.orderStatus.status)
        
    def order_modify(self, trade):
        print(trade.orderStatus.status)
        
events = all_events()
        
ib = IB()


ib.accountSummaryEvent += events.account_update
ib.connectedEvent += events.connected
ib.disconnectedEvent += events.disconnected
# ib.updatePortfolioEvent += events.portfolio_update
ib.positionEvent  += events.position_update
ib.openOrderEvent += events.order_status
ib.orderModifyEvent += events.order_modify
ib.cancelOrderEvent += events.canceled_order
ib.newOrderEvent += events.open_orders




ib.connect('127.0.0.1', 7497, 57)
ib.disconnect()




ES = Future(symbol='ES', lastTradeDateOrContractMonth='20201218', exchange='GLOBEX',
                  currency='USD')
ib.qualifyContracts(ES)
order = LimitOrder("BUY", 2, 3314)
trade = ib.placeOrder(ES, order)
ib.cancelOrder(order)
