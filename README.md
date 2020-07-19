#############

# to use this python program, you should have IBKR account and must be subscribed for future and future options data

#############

# ES-mini and its technical analysis are loaded to a Q-learning agent that train to choose the parameters to maximize 
# and decide to start or end trades. 
#
# The agent backtest with an imaginary account starts with 20K and desiplays end account value.
# For fast results, it is recomended to use GPU or colab notebook to run this code. 

# Note: this is an openAI Gym code. using Q-learning and Replay buffer. 

#The replay buffer class was taken from the udemy class of lazyprogrammer about tensorflow class. 

#The environment and agent class were in part taken from the same code for Deep Q learning lecture with some modifications to adjust the learning for Techanical Analysis and options prices. 

#The function name for deep learning as mlp is the same as the code from the lazey programmer but it is modfied to lstms and dense layers. 

#The epsilon equeation used the logarithm different from the original code. 

#The folders to save agents and scaler were part from the lazyprogrammer code.

#Lazy programmer code is:

#https://github.com/lazyprogrammer/machine_learning_examples/blob/master/tf2.0/rl_trader.py
