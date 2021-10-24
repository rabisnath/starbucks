import datetime
import multiprocessing

from data import *
from models import PCA_risk_model
from backtesting import *

'''
Questions:
    1. What interval size is the best to use?
    2. What should alpha be for the adfuller test?
    3. n_components for PCA, the s_i's, inference window size need to be optimized

    first expt: one with no adfuller test, a group with different values of alpha, other params fixed
    second expt: do no adfuller, different time scales
    third expt: pick ranges of values for n_components, s_buy, s_close, s_close_posn, inference window size
                do one test for each element of the cartesian product

'''

symbols = highest_volume_coins(100)

# Experiment 1
# Investigating how many trades are being filtered by the ADF test

Expt_1_Start = datetime.datetime(2020, 1, 1) 
Expt_1_End = None

Expt_1_1_PCA_Settings = {
    'n_components': 10,
    'use_ad_fuller': False,
    'ad_fuller_alpha': 0.05
}

Expt_1_2_PCA_Settings = {
    'n_components': 10,
    'use_ad_fuller': True,
    'ad_fuller_alpha': 0.01
}

Expt_1_3_PCA_Settings = {
    'n_components': 10,
    'use_ad_fuller': True,
    'ad_fuller_alpha': 0.05
}

Expt_1_4_PCA_Settings = {
    'n_components': 10,
    'use_ad_fuller': True,
    'ad_fuller_alpha': 0.25
}

Expt_1_BT_Settings = {
    's_sell': 2.5,
    's_buy': -2.5,
    's_close_posn': 0.5,
    'enable_shorting': True,
    'inference_window_size': 30,
    'bet_size': 0.05,
    'verbosity': 0
}


def Experiment_1(save_dir='', make_dir=True):

    print("Running Experiment 1")
    print("Getting data for experiment 1")
    prices = price_histories(symbols, '1d', Expt_1_Start, Expt_1_End)
    price_matrix = price_matrix_from_dict(prices)

    print("Running backtest 1/4")
    log_1 = backtest(symbols, price_matrix, PCA_risk_model, Expt_1_1_PCA_Settings, **Expt_1_BT_Settings)
    print("Running backtest 2/4")
    log_2 = backtest(symbols, price_matrix, PCA_risk_model, Expt_1_2_PCA_Settings, **Expt_1_BT_Settings)
    print("Running backtest 3/4")
    log_3 = backtest(symbols, price_matrix, PCA_risk_model, Expt_1_3_PCA_Settings, **Expt_1_BT_Settings)
    print("Running backtest 4/4")
    log_4 = backtest(symbols, price_matrix, PCA_risk_model, Expt_1_4_PCA_Settings, **Expt_1_BT_Settings)

    logs = {
        'Expt_1_1': log_1,
        'Expt_1_2': log_2,
        'Expt_1_3': log_3,
        'Expt_1_4': log_4
    }

    comparison_chart = compare_logs(logs)

    if make_dir:
        try:
            os.mkdir(save_dir)
        except:
            pass

    for label, log in logs.items():
        with open(save_dir+'/'+label+'_results.txt', 'w+') as outfile:
            json.dump(log, outfile, indent=4)
    
    with open(save_dir+'/'+'Expt_1_comparison.csv', 'w+') as outfile:
        comparison_chart.to_csv(outfile)
    
    return comparison_chart


# Experiment 2

Expt_2_PCA_Settings = {
    'n_components': 10,
    'use_ad_fuller': True,
    'ad_fuller_alpha': 0.05
}

Expt_2_BT_Settings = {
    's_sell': 2.5,
    's_buy': -2.5,
    's_close_posn': 0.5,
    'enable_shorting': True,
    'inference_window_size': 30,
    'bet_size': 0.05,
    'verbosity': 0
}

Expt_2_Start = datetime.datetime(2020, 1, 1) 
Expt_2_End = None

intervals_to_test = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1mo']

def Experiment_2(save_dir='', make_dir=True):

    print("Running Experiment 2")

    logs = {}

    for i in intervals_to_test:
        try:
            trial_name = 'Expt_2_{}_intervals'.format(i)
            print("Getting data for {}-intervals ({}/{})".format(i, intervals_to_test.index(i), len(intervals_to_test)))
            prices = price_histories(symbols, i, Expt_2_Start, Expt_2_End)
            price_matrix = price_matrix_from_dict(prices)
            print("Running backtest for {}-intervals ({}/{})".format(i, intervals_to_test.index(i), len(intervals_to_test)))
            log = backtest(symbols, price_matrix, PCA_risk_model, Expt_2_PCA_Settings, **Expt_2_BT_Settings)
            logs[trial_name] = log
        except Exception as e:
            print("Error encountered in Experiment 2, trial for {}-intervals:\n {}".format(i, e))
            
        
    comparison_chart = compare_logs(logs)

    if make_dir:
        try:
            os.mkdir(save_dir)
        except:
            pass

    for label, log in logs.items():
        with open(save_dir+'/'+label+'_results.txt', 'w+') as outfile:
            json.dump(log, outfile, indent=4)
    
    with open(save_dir+'/'+'Expt_2_comparison.csv', 'w+') as outfile:
        comparison_chart.to_csv(outfile)
    
    return comparison_chart


if __name__ == '__main__':
    main_save_dir = 'backtesting_results/'
    try:
        os.mkdir(main_save_dir)
    except:
        pass

    #Experiment_1(save_dir=main_save_dir+'Expt_1')
    #Experiment_2(save_dir=main_save_dir+'Expt_2')

    p1 = multiprocessing.Process(target=Experiment_1, args=(main_save_dir+'Expt_1',))
    p2 = multiprocessing.Process(target=Experiment_2, args=(main_save_dir+'Expt_2',))

    p1.start()
    p2.start()
  
    p1.join()
    p2.join()




    