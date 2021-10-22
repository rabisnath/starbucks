from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

api_key = os.environ.get('binance_key')
api_secret = os.environ.get('binance_secret')
client = Client(api_key, api_secret)

def highest_volume_coins(N, base='BTC'):
    '''
    returns N tickers representing the highest volume coins
    of those with the base str in the ticker
    '''
    info = client.get_all_tickers()
    all_symbols = [d['symbol'] for d in info]
    priced_in_btc = [s for s in all_symbols if 'BTC' in s]

    tickers = client.get_orderbook_tickers()
    tickers = pd.DataFrame.from_dict(tickers)
    col_types = {
        'symbol': 'str',
        'bidPrice': 'float',
        'bidQty': 'float',
        'askPrice': 'float',
        'askQty': 'float',
    }
    tickers = tickers.astype(col_types)
    tickers['totalQty'] = tickers['bidQty'] + tickers['askQty']

    high_volume_coins = tickers[tickers['symbol'].isin(priced_in_btc)].nlargest(N, 'totalQty')
    symbols = high_volume_coins['symbol'].values

    return symbols


def price_histories(symbols, interval_size, start_datetime, end_datetime=None):
    '''
    gets the Open-High-Low-Close-Vol price data for the:
        tickers in the array symbols
        interval size \in {1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1mo}
        datetime objects respresenting the beginning and end of the time window
        (None for end_datetime gives data from start to the present)

    returns a dict, indexed by symbol
    the values of the dict are data tables with OHLCV entries for each point in time
    time increases as row index increases
    '''

    accepted_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1mo']
    if not interval_size in accepted_intervals:
        raise ValueError("Given kline interval size, {}, not supported".format(interval_size))
    
    interval_dict = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1mo': Client.KLINE_INTERVAL_1MONTH,
    }
    
    
    price_histories = {}

    for s in symbols:
        start = str(start_datetime.timestamp())
        end = None
        if end_datetime:
            end = str(end_datetime.timestamp())
        klines = client.get_historical_klines(s, interval_dict[interval_size], start, end)
        klines = [(lambda l: [float(x) for x in l])(candle) for candle in klines]
        history = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'num_trades', 'taker_base_asset_vol', 'taker_quote_asset_vol', 'ignore'])
        price_histories[s] = history

    
    for symbol, df in price_histories.items():
        df['symbol'] = [symbol for i in range(len(df.index))]
        df['time (in units of {})'.format(interval_size)] = [i for i in range(len(df.index))]

    return price_histories

def price_matrix_from_dict(price_histories, time_units=None, column='close'):
    '''
    takes a dict, price_histories, 
        where the key is a symbol, and the value is a time-indexed dataframe
    as well as an integer number of time units to include
        where the subset is taken from the end of the interval
        time_units=None will use all the data
    also takes an optional column header to pull data from
        alternatives include 'open', 'high' and 'low'

    returns an N by M matrix of coin prices, where the 1st coord is time, and 
        the second coord is the coin
    '''

    N = time_units
    if time_units is None:
        N = max([len(price_histories[s][column].values) for s in price_histories.keys()])
    M = len(price_histories.keys())

    price_matrix = np.ones([M, N])

    i = 0
    for s in price_histories.keys():
        history = price_histories[s]
        col = history[column].values
        if len(col) < N:
            print("Insufficient data found for symbol {}".format(s))
            continue
        price_matrix[i] = col
        i += 1

    price_matrix = np.transpose(price_matrix)

    shape = price_matrix.shape
    print("price matrix computed with {} points in time, {} coins".format(shape[0], shape[1]))

    return price_matrix


def returns_from_prices(price_matrix):
    '''
    takes a matrix of prices where the first dim is time, 
        and the second dim is which coin
        (reminder, time moves forward with index in this context)
    returns a matrix of returns
        NB: if the input is N by M, the output will be N-1 by M
    '''

    N, M = price_matrix.shape
    return_matrix = np.ones([N-1, M])
    
    for i in range(N-1):
        for j in range(M):
            new_price = price_matrix[i+1][j]
            old_price = price_matrix[i][j]
            return_matrix[i][j] = (new_price - old_price) / old_price

    return return_matrix


