import numpy as np
import datetime
import json
from data import *
from models import PCA_risk_model

def backtest(price_matrix, symbols, model, model_settings, s_sell=2.5, s_buy=-2.5, s_close_posn=0.5, enable_shorting=True, inference_window_size=30, starting_account_balance=10000, bet_size=0.02, verbosity=1):
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
    all_time_low = starting_account_balance
    all_time_high = starting_account_balance
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
                'verbosity': passed_args['verbosity']
            }
    }
    full_log_as_text = ''
    n_trades = 0
    returns_on_trades = []

    while t < t_max:

        if verbosity >= 1: print('Epoch {}/{}'.format(t, t_max))

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
            
        # scorekeeping

        if account_balance < all_time_low:
            all_time_low = account_balance
        elif account_balance > all_time_high:
            all_time_high = account_balance

        t = t + 1

    value_of_open_positions = 0
    for t in trades:
        if t['direction'] == 'Buy':
            value_of_open_positions += t['current_value']
        elif t['direction'] == 'Sell':
            value_of_open_positions -= t['current_value']

    log['starting_account_balance'] = starting_account_balance
    log['account_balance'] = account_balance
    portfolio_value = account_balance + value_of_open_positions
    log['portfolio_value'] = portfolio_value
    log['all_time_low'] = all_time_low
    log['all_time_high'] = all_time_high
    log['n_trades'] = n_trades
    log['returns_on_trades'] = returns_on_trades
    if n_trades != 0:
        log['win_rate'] = len([x for x in returns_on_trades if x > 0]) / n_trades
    log['std_of_returns'] = np.std(returns_on_trades)
    log['total_return'] = portfolio_value/starting_account_balance - 1
    if log['std_of_returns'] != 0:
        log['sharpe_ratio_wrt_zero'] = log['total_return'] / log['std_of_returns']
        log['sharpe_ratio_wrt_2percentRFR'] = (log['total_return'] - 0.02) / log['std_of_returns']
    else:
        log['sharpe_ratio_wrt_zero'] = "N/A (Variance of returns is zero)"
        log['sharpe_ratio_wrt_2percentRFR'] = "N/A (Variance of returns is zero)"
    log['full_log_as_text'] = full_log_as_text
    log['last_value_of_t'] = t

    if verbosity >= 2: print(log)

    return log


def continue_backtest_redux(price_matrix, symbols, model, model_settings, log):

    # unpacking backtesting settings from the provided log
    passed_bt_settings = log['backtest_settings']
    s_sell = passed_bt_settings['s_sell']
    s_buy = passed_bt_settings['s_buy']
    s_close_posn = passed_bt_settings['s_close_posn']
    enable_shorting = passed_bt_settings['enable_shorting']
    inference_window_size = passed_bt_settings['inference_window_size']
    starting_account_balance = passed_bt_settings['starting_account_balance']
    bet_size = passed_bt_settings['bet_size']
    verbosity = passed_bt_settings['verbosity']

    i = 0
    i_max = price_matrix.shape[0] - inference_window_size
    account_balance = log['account_balance']
    trades = [] # trade has {symbol: , direction: , size: , entry_time: }

    n_trades = 0
    returns_on_trades = []

    while i < i_max:
        t = log['last_value_of_t'] + i
        if verbosity >= 1: print('Epoch {}/{}. t = {}.'.format(i, i_max, t))

        log['full_log_as_text'] += 't = {}\n'.format(t)
        log['full_log_as_text'] += 'current open positions: ' + str(trades) + '\n'

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
            
        # scorekeeping

        if account_balance < log['all_time_low']:
            log['all_time_low'] = account_balance
        elif account_balance > log['all_time_high']:
            log['all_time_high'] = account_balance

        i += 1

    value_of_open_positions = 0
    for t in trades:
        if t['direction'] == 'Buy':
            value_of_open_positions += t['current_value']
        elif t['direction'] == 'Sell':
            value_of_open_positions -= t['current_value']

    log['account_balance'] = account_balance
    portfolio_value = account_balance + value_of_open_positions
    log['portfolio_value'] = portfolio_value
    log['n_trades'] += n_trades
    log['returns_on_trades'].extend(returns_on_trades) 
    if log['n_trades'] != 0:
        log['win_rate'] = len([x for x in log['returns_on_trades'] if x > 0]) / log['n_trades']
    log['std_of_returns'] = np.std(log['returns_on_trades'])
    log['total_return'] = portfolio_value/starting_account_balance - 1
    if log['std_of_returns'] != 0:
        log['sharpe_ratio_wrt_zero'] = log['total_return'] / log['std_of_returns']
        log['sharpe_ratio_wrt_2percentRFR'] = (log['total_return'] - 0.02) / log['std_of_returns']
    else:
        log['sharpe_ratio_wrt_zero'] = "N/A (Variance of returns is zero)"
        log['sharpe_ratio_wrt_2percentRFR'] = "N/A (Variance of returns is zero)"
    log['last_value_of_t'] += i

    if verbosity >= 2: print(log)


    return

def continue_backtest(log, price_matrix):
    '''
    performs a backtest using settings from a previous backtest as a starting point instead of initializing everything from scratch
    '''

    # unpacking backtesting settings from the provided log
    passed_bt_settings = log['backtest_settings']
    symbols = passed_bt_settings['symbols']
    model = passed_bt_settings['model']
    model_settings = passed_bt_settings['model_settings']
    s_sell = passed_bt_settings['s_sell']
    s_buy = passed_bt_settings['s_buy']
    s_close_posn = passed_bt_settings['s_close_posn']
    enable_shorting = passed_bt_settings['enable_shorting']
    inference_window_size = passed_bt_settings['inference_window_size']
    starting_account_balance = passed_bt_settings['starting_account_balance']
    bet_size = passed_bt_settings['bet_size']
    verbosity = passed_bt_settings['verbosity']

    # setting the backtest on the given price matrix
    t = 0
    t_max = price_matrix.shape[0] - inference_window_size
    global_t = t + log['last_value_of_t']
    account_balance = log['account_balance']
    all_time_low = log['all_time_low']
    all_time_high = log['all_time_high']
    trades = [] # trade has {symbol: , direction: , size: , entry_time: }
    full_log_as_text = ''
    n_trades = 0
    returns_on_trades = []

    while t < t_max:

        if verbosity >= 1: 
            print('Epoch {} since start of backtest; {}/{} for the current data chunk'.format(global_t, t, t_max))

        full_log_as_text += 't = {}\n'.format(global_t)
        full_log_as_text += 'current open positions: ' + str(trades) + '\n'
        log['Open positions at time t={}'.format(global_t)] = trades

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
            
        # scorekeeping

        if account_balance < all_time_low:
            all_time_low = account_balance
        elif account_balance > all_time_high:
            all_time_high = account_balance

        t += 1
        global_t += 1

    value_of_open_positions = log['final_portfolio_value'] - account_balance
    for t in trades:
        if t['direction'] == 'Buy':
            value_of_open_positions += t['current_value']
        elif t['direction'] == 'Sell':
            value_of_open_positions -= t['current_value']

    log['starting_account_balance'] = starting_account_balance
    log['account_balance'] = account_balance
    final_portfolio_value = account_balance + value_of_open_positions
    log['final_portfolio_value'] = final_portfolio_value
    log['all_time_low'] = all_time_low
    log['all_time_high'] = all_time_high
    log['n_trades'] += n_trades
    log['returns_on_trades'].extend(returns_on_trades)
    if n_trades != 0:
        log['win_rate'] = len([x for x in returns_on_trades if x > 0]) / n_trades
    log['std_of_returns'] = np.std(returns_on_trades)
    log['total_return'] = final_portfolio_value/starting_account_balance - 1
    if log['std_of_returns'] != 0:
        log['sharpe_ratio_wrt_zero'] = log['total_return'] / log['std_of_returns']
        log['sharpe_ratio_wrt_2percentRFR'] = (log['total_return'] - 0.02) / log['std_of_returns']
    else:
        log['sharpe_ratio_wrt_zero'] = "N/A (Variance of returns is zero)"
        log['sharpe_ratio_wrt_2percentRFR'] = "N/A (Variance of returns is zero)"
    log['full_log_as_text'] = full_log_as_text
    log['last_value_of_t'] = global_t

    if verbosity >= 2: print(log)

    return log


def backtest_in_chunks(price_matrices, symbols, model, model_settings, s_sell=2.5, s_buy=-2.5, s_close_posn=0.5, enable_shorting=True, inference_window_size=30, starting_account_balance=10000, bet_size=0.02, verbosity=1):
    '''
    takes all the same settings as backtest(), but takes a list of price matrices to backtest on sequentially 
    instead of a single large matrix
    '''

    # if prices matrices is a list containing a single price matrix, return normal backtest of that matrix
    # elif there's more than one, run normal backtest on first one, then continue_backtest on the others

    #continue_backtest_redux(price_matrix, symbols, model, model_settings, log)
    if price_matrices == []:
        raise ValueError("Received empty list of matrices")
    elif len(price_matrices) == 1:
        log = backtest(price_matrices[0], symbols, model, model_settings, s_sell=s_sell, s_buy=s_buy, s_close_posn=s_close_posn, enable_shorting=enable_shorting, inference_window_size=inference_window_size, starting_account_balance=starting_account_balance, bet_size=bet_size, verbosity=verbosity)
        return log
    elif len(price_matrices) > 1:
        log = backtest(price_matrices[0], symbols, model, model_settings, s_sell=s_sell, s_buy=s_buy, s_close_posn=s_close_posn, enable_shorting=enable_shorting, inference_window_size=inference_window_size, starting_account_balance=starting_account_balance, bet_size=bet_size, verbosity=verbosity)
        i = 1
        while i < len(price_matrices):
            continue_backtest_redux(price_matrices[i], symbols, model, model_settings, log)
            i += 1
        return log

    return


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

    symbols = highest_volume_coins(100)

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
        'verbosity': 1
    }

    # a sample backtest
    
    print("\nTesting simple backtest...\n")

    bt_start = datetime.datetime(2021, 1, 1)
    bt_end = datetime.datetime(2021, 12, 1)
    prices = price_histories(symbols, '1d', bt_start, bt_end)
    price_matrix = price_matrix_from_dict(prices)

    log = backtest(price_matrix, symbols, PCA_risk_model, example_PCA_settings, **example_backtest_settings)
    with open('backtest_results.txt', 'w+') as outfile:
        json.dump(log, outfile, indent=4)
    
    # testing the backtest_in_chunks method with one chunk

    print("\nTesting backtest_in_chunks with one chunk...\n")

    bt_start = datetime.datetime(2021, 1, 1)
    bt_end = datetime.datetime(2021, 12, 1)
    prices = price_histories(symbols, '1d', bt_start, bt_end)
    price_matrix = price_matrix_from_dict(prices)

    log = backtest_in_chunks([price_matrix], symbols, PCA_risk_model, example_PCA_settings, **example_backtest_settings)
    with open('backtest_chunks_1_results.txt', 'w+') as outfile:
        json.dump(log, outfile, indent=4)

    # testing backtest_in_chunks with more than one chunk

    print("\nTesting backtest_in_chunks with multiple chunks...\n")

    start = datetime.datetime(2021, 1, 1)
    end = start + datetime.timedelta(days=360)
    delta = datetime.timedelta(days=30)
    print("Getting daily data from {} to {} in chunks of size {}".format(start, end, delta))
    price_matrices = get_data_as_chunks(symbols, '1d', get_date_pairs(start, end, delta))

    log = backtest_in_chunks(price_matrices, symbols, PCA_risk_model, example_PCA_settings, **example_backtest_settings)
    with open('backtest_chunks_3_results.txt', 'w+') as outfile:
        json.dump(log, outfile, indent=4)

