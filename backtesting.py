import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import json

from data import *
from models import PCA_risk_model

def backtest(symbols, price_matrix, model, model_settings, s_sell=2.5, s_buy=-2.5, s_close_posn=0.5, enable_shorting=True, inference_window_size=30, starting_account_balance=10000, bet_size=0.02, verbosity=1):
    '''
    the backtest function takes a model and simulates its performance using the provided data
    
    required parameters:

    start - datetime object with the starting date/time for the simulation
                if the data provided has, say, daily frequency, then it only makes sense to specify the start up to the day 
                e.g. start = datetime.datetime(2020, 1, 1)
    end - datetime object specifying the end of the simulation
    symbols - a list of strings with the tickers for all the coins/stocks reperesented in the price_matrix
    price_matrix - an N by M matrix of coin prices, where the 1st coord is time, and the second coord is the coin
    model - 
        a function that takes a matrix of returns^* as its sole required arg, can take any number of kwargs
        returns a list of floats that will be interpreted as 's_scores', a trading signal is generated from interpreting this as the z-score of a given coin's returns
            *(this fn will make the step of turning prices into returns for each inference_window_size'd slice of the price_matrix)
    model_settings - a dict of key word args to pass to the model

    optional parameters:

    s_sell - sell if s > s_sell
    s_buy - buy if s < s_buy
    s_close_posn - unwind current position if |s| < s_close_posn
    enable_shorting - if false, will only act on buy signals
    inference_window_size - the number of rows of the price_matrix that will be used to make decisions at each simulated point in time
                            the length of the sim is therefore len(price_matrix) - inference_window_size
    starting_account_balance - amount of cash to begin the simluation with (in the same units of the prices given)
    bet_size - trades will be placed of size current_cash * bet_size
    verbosity - if verbosity==0, print nothing, if verbosity==1, print log at the end

    '''

    t = 0
    t_max = price_matrix.shape[0] - inference_window_size
    account_balance = starting_account_balance
    trades = [] # trade has {symbol: , direction: , size: , entry_time: }
    log = {}
    full_log_as_text = ''

    while t < t_max:

        if verbosity >= 1: print('Epoch: {}/{}'.format(t, t_max))

        full_log_as_text += 't = {}\n'.format(t)
        full_log_as_text += 'current open positions: ' + str(trades) + '\n'

        log[t] = trades

        # new day
        # get data slice
        # call pca and get s scores
        # iterate over s scores and generate trades
        # simulate trades in portfolio (including flash forward)

        data_slice = price_matrix[t: t+inference_window_size]
        return_slice = returns_from_prices(data_slice)
        s_scores = model(return_slice, **model_settings)
        s_score_dict = {}
        for i, symbol in enumerate(symbols):
            s_score_dict[symbol] = s_scores[i]

        
        # updating trade values with return data
        # unwinding trades

        new_trades = []
        for trade in trades:
            trade['value'] *= (1 + return_slice[-1, list(symbols).index(trade['symbol'])])
            if s_score_dict[trade['symbol']]**2 < s_close_posn**2:
                account_balance += np.abs(trade['value'])
            else:
                new_trades.append(trade)
        trades = new_trades
        
        
        # making new trades based on signals

        for symbol, score in s_score_dict.items():
            bet = account_balance * bet_size
            if account_balance > 0:
                if score < s_buy:
                    trades.append({'symbol': symbol, 'direction': 'Buy', 'value': bet, 'entry_time': t})
                    account_balance -= bet
                if (score > s_sell) and enable_shorting: 
                    trades.append({'symbol': symbol, 'direction': 'Sell', 'value': bet, 'entry_time': t})
                    account_balance += bet
            

        t = t + 1

    log['full_log_as_text'] = full_log_as_text
    log['starting_account_balance'] = starting_account_balance
    log['final_account_balance'] = account_balance

    if verbosity >= 1: print(log)

    return log


if __name__ == '__main__':

    example_PCA_settings = {
        'n_components': 10,
        'use_ad_fuller': True,
        'ad_fuller_alpha': 0.05
    }

    example_backtest_settings = {
        's_sell': 2.5,
        's_buy': -2.5,
        's_close_posn': 0.5,
        'enable_shorting': False,
        'inference_window_size': 30,
        'bet_size': 0.05,
        'verbosity': 0
    }

    bt_start = datetime.datetime(2020, 11, 1) 
    bt_end = datetime.datetime(2021, 1, 1)
    symbols = highest_volume_coins(100)
    prices = price_histories(symbols, '1d', bt_start, bt_end)
    price_matrix = price_matrix_from_dict(prices)

    log = backtest(symbols, price_matrix, PCA_risk_model, example_PCA_settings, **example_backtest_settings)
    with open('backtest_results.txt', 'w+') as outfile:
        json.dump(log, outfile)

