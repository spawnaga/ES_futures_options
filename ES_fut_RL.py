# -*- coding: utf-8 -*-
"""
Created on Sat May 30 22:45:37 2020

@author: alial
"""


import numpy as np
import pandas as pd



from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import nest_asyncio
from datetime import datetime, timedelta
import itertools
import os
import pickle
import math
from sklearn.preprocessing import StandardScaler
from ressup import ressup
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
nest_asyncio.apply()

def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

class get_data:

    def next_exp_weekday(self):
        weekdays = {2: [5, 6, 0], 4: [0, 1, 2], 1: [3, 4]}
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
        ES = Future(symbol='ES', lastTradeDateOrContractMonth='20201218', exchange='GLOBEX',
                                currency='USD')
        ib.qualifyContracts(ES)
        ES_df = ib.reqHistoricalData(contract=ES, endDateTime=endDateTime, durationStr=No_days,
                                     barSizeSetting=interval, whatToShow = 'TRADES', useRTH = False)
        ES_df = util.df(ES_df)
        ES_df.set_index('date',inplace=True)
        ES_df.index = pd.to_datetime(ES_df.index)
        ES_df['hours'] = ES_df.index.strftime('%H').astype(int)
        ES_df['minutes'] = ES_df.index.strftime('%M').astype(int)
        ES_df['hours + minutes'] = ES_df['hours']*100 + ES_df['minutes']
        ES_df['Day_of_week'] = ES_df.index.dayofweek
        ES_df['Resistance'], ES_df['Support'] = self.res_sup(ES_df)
        ES_df['RSI'] = ta.RSI(ES_df['close'])
        ES_df['macd'],ES_df['macdsignal'],ES_df['macdhist'] = ta.MACD(ES_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        ES_df['macd - macdsignal'] = ES_df['macd'] - ES_df['macdsignal']
        ES_df['MA_9']=ta.MA(ES_df['close'], timeperiod=9)
        ES_df['MA_26']=ta.MA(ES_df['close'], timeperiod=26)
        ES_df['MA_200']=ta.MA(ES_df['close'], timeperiod=200)
        ES_df['EMA_9']=ta.EMA(ES_df['close'], timeperiod=9)
        ES_df['EMA_26']=ta.EMA(ES_df['close'], timeperiod=26)
        ES_df['EMA_50']=ta.EMA(ES_df['close'], timeperiod=50)
        ES_df['EMA_200']=ta.EMA(ES_df['close'], timeperiod=200)
        ES_df['ATR']=ta.ATR(ES_df['high'],ES_df['low'], ES_df['close'])
        ES_df['roll_max_cp']=ES_df['high'].rolling(20).max()
        ES_df['roll_min_cp']=ES_df['low'].rolling(20).min()
        ES_df['roll_max_vol']=ES_df['volume'].rolling(20).max()
        ES_df['vol/max_vol'] = ES_df['volume']/ES_df['roll_max_vol']
        ES_df['EMA_9-EMA_26']=ES_df['EMA_9']-ES_df['EMA_26']
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

# =============================================================================
# start machine learning some parts of this comming code were taken from 
# the lazy programmer course in udemy for Tensorflow machine learning class
# Lecture of Q-Learning 
# https://github.com/lazyprogrammer/machine_learning_examples/blob/master/tf2.0/rl_trader.py
# =============================================================================

### The experience replay memory ###




def get_scaler(env):
      # return scikit-learn scaler object to scale the states
      # Note: you could also populate the replay buffer here
    
      states = []
      for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
          break
    
      scaler = StandardScaler()
      scaler.fit(states)
      return scaler




def maybe_make_dir(directory):
      if not os.path.exists(directory):
          os.makedirs(directory)
    



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


class ReplayBuffer:
      def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size
    
      def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    
      def sample_batch(self, batch_size=50):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])

class MultiStockEnv:
    """
      A 3-stock trading environment.
      State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
    - cash owned (can be used to purchase more stocks)
      Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
      """
    def __init__(self, data, initial_investment=20000):
        # data
        self.stock_price_history = data.iloc[:,-2:].values
        self.n_step, self.state_dim = data.shape
        self.atr=data_raw.loc[:,'ATR']
        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.n_stock = 2
        
        self.action_space = np.arange(3**self.n_stock)
        
        # action permutations
        # returns a nested list with elements like:
        # [0,0]
        # [1,0]
        # [0,1]
        # [1,1]
        # [0,2]
        # etc.
        # 0 = sell
        # 1 = hold
        # 2 = buy
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
    
        # calculate size of state
        self.state_dim = self.state_dim + 3
        
        self.reset()
        
    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price
        obs[4] = self.cash_in_hand
        obs[5:] = data.iloc[self.cur_step,:-2]
        return obs
    
    def _get_val(self):
        return self.stock_owned.dot(self.stock_price*50) + self.cash_in_hand
        
    def _trade(self, action):
      # index the action we want to perform
      # 0 = sell
      # 1 = hold
      # 2 = buy
      # e.g. [2,1,0] means:
      # buy first stock
      # hold second stock
      # sell third stock
      action_vec = self.action_list[action]
      
      # determine which stocks to buy or sell
      sell_index = [] # stores index of stocks we want to sell
      buy_index = [] # stores index of stocks we want to buy
      for i, a in enumerate(action_vec):
        if a == 0:
          sell_index.append(i)
        elif a == 2:
          buy_index.append(i)
      # print(data_raw["low"].iloc[self.cur_step], data_raw["close"].iloc[self.cur_step-1], 0.75 * data_raw["ATR"].iloc[self.cur_step-1], data_raw["ATR"].iloc[self.cur_step-3], self.stock_owned[0], sell_index)
      if data_raw["low"].iloc[self.cur_step] < data_raw["close"].iloc[self.cur_step-1] - (2 * data_raw["ATR"].iloc[self.cur_step-1] if data_raw["ATR"].iloc[self.cur_step-1] > data_raw["ATR"].iloc[self.cur_step-3] else data_raw["ATR"].iloc[self.cur_step-3] ) and self.stock_owned[0] !=0 and sell_index==[]:
          sell_index.append(0)
    
      elif data_raw["high"].iloc[self.cur_step] >  data_raw["close"].iloc[self.cur_step-1] + (2 * data_raw["ATR"].iloc[self.cur_step-1] if data_raw["ATR"].iloc[self.cur_step-1] > data_raw["ATR"].iloc[self.cur_step-3] else data_raw["ATR"].iloc[self.cur_step-3] ) and self.stock_owned[1] !=0 and sell_index==[]:
          sell_index.append(1)
      # sell any stocks we want to sellself.stock_owned[1]
      # then buy any stocks we want to buy
      if sell_index:
        # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
        for i in sell_index:
          self.cash_in_hand += self.stock_price[i] * self.stock_owned[i] * 50
          self.stock_owned[i] = 0
      if buy_index:
        # NOTE: when buying, we will loop through each stock we want to buy,
        #       and buy one share at a time until we run out of cash
        can_buy = True
        while can_buy:
          for i in buy_index:
            if self.cash_in_hand > self.stock_price[i] * 50:
              self.stock_owned[i] += 1 # buy one share
              self.cash_in_hand -= self.stock_price[i] * 50
              # print(f'price = {self.stock_price[i]}, cash in hand={self.cash_in_hand}, no of stocks = {self.stock_owned[i]}')
            else:
              can_buy = False

    def step(self, action):
        assert action in self.action_space
    
        # get current value before performing the action
        prev_val = self._get_val()
    
        # update price, i.e. go to the next day
        self.cur_step += 1

        self.stock_price = self.stock_price_history[self.cur_step]
    
        # perform the trade
        self._trade(action)
    
        # get the new value after taking the action
        cur_val = self._get_val()
    
        # reward is the increase in porfoliosignificantly different from zero value
        reward = cur_val - prev_val
    
        # done if we have run out of data
        done = self.cur_step == self.n_step - 1
    
        # store the current value of the portfolio here
        info = {'cur_val': cur_val}
    
        # conform to the Gym API
        return self._get_obs(), reward, done, info

class DQNAgent(object):
    def __init__(self, state_size, action_size):
      self.state_size = state_size
      self.action_size = action_size
      self.memory = ReplayBuffer(state_size, action_size, size=1500)
      self.gamma = 0.97  # discount rate
      # self.epsilon = 1 # exploration rate
      self.epsilon_min = 0.2
      self.epsilon_decay = 0.001
      self.model = mlp(state_size, action_size)
      self.random_trades = 0

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
    
    
    def act(self, state):
      if np.random.rand() <= self.epsilon:
          self.random_trades +=1
          return np.random.choice(self.action_size)
      act_values = self.model.predict(state)
      return np.argmax(act_values[0])  # returns action
    
    
    def replay(self, batch_size=32):
      # first check if replay buffer contains enough data
      if self.memory.size < batch_size:
            return

      # sample a batch of data from the replay memory
      minibatch = self.memory.sample_batch(batch_size)
      states = minibatch['s']
      actions = minibatch['a']
      rewards = minibatch['r']
      next_states = minibatch['s2']
      done = minibatch['d']
    
      # Calculate the tentative target: Q(s',a)
      target = rewards + (1 - done) * self.gamma * np.amax(self.model.predict(next_states), axis=1)
    
      # With the Keras API, the target (usually) must have the same
      # shape as the predictions.
      # However, we only need to update the network for the actions
      # which were actually taken.
      # We can accomplish this by setting the target to be equal to
      # the prediction for all values.
      # Then, only change the targets for the actions taken.
      # Q(s,a)
      target[done] = rewards[done]
      
      
      target_full = self.model.predict(states)

      target_full[np.arange(batch_size), actions] = target
    
      # Run one training step
      self.model.train_on_batch(states, target_full)
    
      if self.epsilon > self.epsilon_min:
        self.epsilon = self.epsilon_min + (self.epsilon) * \
          math.exp(-1 * env.cur_step * self.epsilon_decay)

    
    def load(self, name):
        self.model.load_weights(name)
    
    
    def save(self, name):
        self.model.save_weights(name)

def play_one_episode(agent, env):
    # note: after transforming states are already 1xD
    state = env.reset()
    original = state
    state = scaler.transform([state])
    done = False
    agent.random_trades = 0
    old_action = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        if (original[:2]!= next_state[:2]).any() :
            print(f'action = {action}, actiontype = {env.action_list[action]}, reward = {reward}, end_value = {info["cur_val"]}')
            print(f'holding calls = {next_state[0]} , puts = {next_state[1]} and action = {action} reward = {reward}')
        original = next_state
        old_action = action
        next_state = scaler.transform([next_state])
        agent.update_replay_memory(state, action, reward, next_state, done)
        agent.replay(batch_size)
        state = next_state

    
    return info['cur_val'] #if info['cur_val'] > 20000 else 0

def test_trade(agent, env):
    state = env.reset()
    original = state
    state = scaler.transform([state])
    done = False
    while not done:
        action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        
        if (original[:2]!= next_state[:2]).any() :
            print(f'holding calls = {next_state[0]} , puts = {next_state[1]} and action = {action} reward = {reward}')
        original = next_state
        next_state = scaler.transform([next_state])
        
        state = next_state
    return info['cur_val']

if __name__ == '__main__':
    import os

    path = os.getcwd() 
   
    # config
    models_folder = f'{path}/rl_trader_models_Sup/1_layer_BO_RSI_ATR_Close' #where models and scaler are saved
    rewards_folder = f'{path}/rl_trader_rewards_Sup/1_layer_BO_RSI_ATR_Close' #where results are saved
    num_episodes = 10 #number of loops per a cycle
    
    initial_investment = 4000


    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)
    
    res=get_data()
    use = 'train' #define the use for the code train or test

   
    while True:
        
        succeded_trades = 0 # To count percentage of success
        try:
            from ib_insync import *
            import talib as ta
            from talib import MA_Type
            ib = IB()
            ib.connect('104.237.11.181', 7497, clientId=np.random.randint(10, 1000))
            ES = Future(symbol='ES', lastTradeDateOrContractMonth='20201218', exchange='GLOBEX',
                        currency='USD')
            ib.qualifyContracts(ES)
            endDateTime = ''
            No_days = '3 D'
            interval = '1 min'
            data_raw = res.options(res.options(res.ES(),res.option_history(res.get_contract('C', 2000)))\
                               ,res.option_history(res.get_contract('P', 2000))) #collect live data of ES with TA and options prices
            data_raw.to_csv('./new_data.csv') # save data incase tws goes dowen
        except:
            data_raw=pd.read_csv('./new_data.csv',index_col='date')

        data = data_raw[['hours + minutes', 'EMA_9-EMA_26', 'EMA_200-EMA_50', 'RSI', 'ATR', 'vol/max_vol', 'ES_C_close','ES_P_close']] #choose parameters to drop if not needed
        n_stocks = 2
        train_data = data
        batch_size = 1000
        env = MultiStockEnv(train_data, initial_investment) # start envirnoment
        state_size = env.state_dim
        action_size = len(env.action_space)
        agent = DQNAgent(state_size, action_size)
        scaler = get_scaler(env)
        agent.epsilon = 5
        try:
            agent.load(f'{models_folder}/dqn.h5') # load agent
        except Exception as error:
            print(error)
        try:
            with open(f'{rewards_folder}/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)  # load scaler  # load scaler
        except Exception as error:
            print(error)
        # store the final value of the portfolio (end of episode)
        portfolio_value = []

        if use == 'train':
            try:
                for e in range(num_episodes):
                    t0 = datetime.now()
                    val = play_one_episode(agent, env)
                    if val >initial_investment: # take only profitable trades
                        succeded_trades +=1
                    print(f'Number of random trades = {agent.random_trades} from {len(train_data)} or {round(100*agent.random_trades/len(train_data),0)}% and Epsilon = {agent.epsilon}' )
                    dt = datetime.now() - t0
                    print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
                    if val/ initial_investment < .8:
                        print('succeded trdes less than 80%. Not pass saving this epsode')
                        continue
                    portfolio_value.append(val) # append episode end portfolio value
                    if agent.epsilon >= agent.epsilon_min:
                        agent.epsilon_min + (agent.epsilon) * \
                            math.exp(-1 * env.cur_step * agent.epsilon_decay)
         
                print(f'*****Loop finished, No. of succeded trades = {succeded_trades}, percentage = {succeded_trades/num_episodes*100}%')
                agent.save(f'{models_folder}/dqn.h5')
                
                
                # save the scaler
                with open(f'{rewards_folder}/scaler.pkl', 'wb') as f:
                  pickle.dump(scaler, f)
                  f.close()
        
                # save portfolio value for each episode
        
                np.save(f'{rewards_folder}/reward.npy', np.array(portfolio_value))
                np.save(f'{rewards_folder}/succeded_trades.npy', np.array(succeded_trades))
                np.save(f'{rewards_folder}/succeded_trades.npy', np.array(agent.random_trades))
            except KeyboardInterrupt:
                print(f'*****Loop finished, No. of succeded trades = {succeded_trades}, percentage = {succeded_trades/(e + 1)*100}%')
                agent.save(f'{models_folder}/dqn.h5')
                
                
                # save the scaler
                with open(f'{rewards_folder}/scaler.pkl', 'wb') as f:
                  pickle.dump(scaler, f)
                  f.close()
        
                # save portfolio value for each episode
        
                np.save(f'{rewards_folder}/reward.npy', np.array(portfolio_value))
                np.save(f'{rewards_folder}/succeded_trades.npy', np.array(succeded_trades))
                np.save(f'{rewards_folder}/succeded_trades.npy', np.array(agent.random_trades))
                break
            except Exception as error:
                print("UNEXPECTED EXCEPTION")
                print(error)
                break
        else:
            agent.epsilon = 0.0001
            t0 = datetime.now()
            val = test_trade(agent, env)
            dt = datetime.now() - t0
            print(f'Number of random trades = {agent.random_trades} from {len(data)} or {round(100*agent.random_trades/len(data),0)}% and Epsilon = {agent.epsilon} and final value={val}' )
            break





