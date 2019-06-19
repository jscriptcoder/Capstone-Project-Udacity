import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
from statsmodels.tsa.stattools import adfuller

def load_csv_with_dates(file):
    return pd.read_csv(file, 
                       index_col=0, 
                       parse_dates=True, 
                       infer_datetime_format=True)

def get_apple_stock(corrected=True):
    apple_stock = load_csv_with_dates('datasets/AAPL_yahoo-finance_19801212-20190531.csv')
    
    # for the sake of simplicity I'm gonna drop Adj Close column
    apple_stock.drop(columns='Adj Close', inplace=True)
    
    if corrected == True:
        apple_stock.loc['1981-08-10'] = (apple_stock.loc['1981-08-07'] + apple_stock.loc['1981-08-11']) / 2
    
    return apple_stock

def get_apple_close_price():
    apple_stock = get_apple_stock()
    return apple_stock['Close']

def get_range(start, end, df):
    return df[(df.index >= start) & (df.index <= end)]

def plot_field_over_time(df, y='Close', xlabel='Year', ylabel=None, ylegend=None, title='', figsize=(15, 6)):
    ax = df.reset_index().plot(x='Date', 
                               y=y, 
                               title=title,
                               figsize=figsize)
    ax.set_xlabel(xlabel)
    
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if ylegend is not None:
        ax.legend([ylegend])

def plot_candlestick(df, xlabel, ylabel, title='', figsize=(15, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    candlestick_ohlc(ax, 
                     zip(mdates.date2num(df.index.to_pydatetime()), 
                         df['Open'], 
                         df['High'], 
                         df['Low'], 
                         df['Close']), 
                     width=0.6, 
                     colorup='g')
    ax.xaxis_date()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

# see: https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
def adf_test(series):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
        'Test Statistic',
        'p-value',
        'Lags Used',
        'Number of Observations Used'
    ])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
        
    print (dfoutput)

def plot_series(series, title='', figsize=(15, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(series)
    ax.set_title(title)

def difference(series, n=1):
    return np.hstack((series[0], np.diff(series)))

def inverse_difference(pre_history_last_value, series):
    return np.hstack((pre_history_last_value, series)).cumsum()[1:]

def log_transform(series):
    return np.log(series)

def inverse_log_transform(series):
    return np.exp(series)

def rmse(preds, targets):
    return np.sqrt(((preds - targets) ** 2).mean())

def plot_walk_forward_validation(test, predictions, size=21, steps=1):
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.plot(test[:size])
    plt.plot(predictions)
    ax.set_title('Walk forward validation - {} days, {} days forecast'.format(size, steps))
    ax.legend(['Expected', 'Predicted'])

def split_sequence(seq, look_back):
    '''
    split a univariate sequence into samples
    seq = [1, 2, 3, 4, 5, 6]
    look_back = 2
    
      X       y
    -------------
    [1, 2]    3
    [3, 4]    5
    [4, 5]    6
        
    '''
    X, y = list(), list()
    seq_len = len(seq)
    
    for i in range(seq_len):
        # find the end of this pattern
        target_i = i + look_back
        
        # check if we are beyond the sequence
        if target_i > seq_len-1: break
        
        # gather input and output parts of the pattern
        seq_x, seq_y = seq[i:target_i], seq[target_i]
        
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)

# Inspired by https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	'''
    Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
    '''
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
    
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
    
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
    
	return agg

def plot_columns(df, columns, fromTo=None, figsize=(15, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    if fromTo is not None:
        df = get_range(fromTo[0], fromTo[1], df)
        
    for column in columns:
        df.reset_index().plot(ax=ax, x='Date', y=column)