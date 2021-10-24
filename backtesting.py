import numpy as np
import datetime
import json
from data import *
from models import PCA_risk_model

def backtest(symbols, price_matrix, model, model_settings, s_sell=2.5, s_buy=-2.5, s_close_posn=0.5, enable_shorting=True, inference_window_size=30, starting_account_balance=10000, bet_size=0.02, verbosity=1):
    '''
    the backtest function takes a model and simulates its performance using the provided data
    
    required parameters:

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
    passed_args = locals()
    log = {
        'backtest_settings': {
                'symbols': str(passed_args['symbols']),
                'model': str(passed_args['model']),
                'model_settings': str(passed_args['model_settings']),
                's_sell': passed_args['s_sell'],
                's_buy': passed_args['s_buy'],
                's_close_posn': passed_args['s_close_posn'],
                'enable_shorting': passed_args['enable_shorting'],
                'inference_window_size': passed_args['inference_window_size'],
                'starting_account_balance': passed_args['starting_account_balance'],
                'bet_size': passed_args['bet_size'],
            }
    }
    full_log_as_text = ''
    n_trades = 0
    returns_on_trades = []

    while t < t_max:

        if verbosity >= 1: print('Epoch: {}/{}'.format(t, t_max))

        full_log_as_text += 't = {}\n'.format(t)
        full_log_as_text += 'current open positions: ' + str(trades) + '\n'

        log['Open positions at time t={}'.format(t)] = trades

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
            trade['current_value'] *= (1 + return_slice[-1, list(symbols).index(trade['symbol'])])
            sign = 1 if trade['direction'] == 'Long' else -1
            trade['profit_and_loss'] = sign * (trade['current_value'] - trade['original_value'])
            if s_score_dict[trade['symbol']]**2 < s_close_posn**2:
                account_balance += sign * np.abs(trade['current_value'])
                returns_on_trades.append((trade['current_value']/trade['original_value']) - 1)
            else:
                new_trades.append(trade)
        trades = new_trades
        
        
        # making new trades based on signals

        for symbol, score in s_score_dict.items():
            bet = account_balance * bet_size
            if account_balance > 0:
                if score < s_buy:
                    trades.append({'symbol': symbol, 'direction': 'Long', 'original_value': bet, 'current_value': bet, 'entry_time': t})
                    account_balance -= bet
                    n_trades += 1
                if (score > s_sell) and enable_shorting: 
                    trades.append({'symbol': symbol, 'direction': 'Short', 'original_value': bet, 'current_value': bet, 'entry_time': t})
                    account_balance += bet
                    n_trades += 1
            

        t = t + 1

    value_of_open_positions = 0
    for t in trades:
        if t['direction'] == 'Buy':
            value_of_open_positions += t['current_value']
        elif t['direction'] == 'Sell':
            value_of_open_positions -= t['current_value']

    log['starting_account_balance'] = starting_account_balance
    log['final_account_balance'] = account_balance
    final_portfolio_value = account_balance + value_of_open_positions
    log['final_portfolio_value'] = final_portfolio_value
    log['n_trades'] = n_trades
    log['returns_on_trades'] = str(returns_on_trades)
    log['total_return'] = final_portfolio_value/starting_account_balance - 1
    log['trading_history_as_text'] = full_log_as_text

    if verbosity >= 1: print(log)

    return log


def compare_logs(logs, settings_to_include=[]):
    '''
    takes a dict of labeled logs and returns a dataframe with summary info
    will only include backtesting settings in the settings_to_include list
    '''

    items_to_extract = settings_to_include + ['n_trades', 'total_return']
    data = [[l[i] for i in items_to_extract] for l in logs.values()]

    df = pd.DataFrame(data, columns=items_to_extract, index=logs.keys())


    return df


if __name__ == '__main__':

    example_PCA_settings = {
        'n_components': 10,
        'use_ad_fuller': False,
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
        json.dump(log, outfile, indent=4)

