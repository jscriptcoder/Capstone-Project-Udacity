import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

def load_csv_with_dates(file):
    '''
    Loads csv files with first column dates
    '''
    return pd.read_csv(file, 
                       index_col=0, 
                       parse_dates=True, 
                       infer_datetime_format=True)

def get_apple_stock(corrected=True):
    '''
    Loads apple stock prices from Yahoo! finance
    params:
        corrected: if True it'll correct the missing row on 1981-08-10
    '''
    apple_stock = load_csv_with_dates('datasets/AAPL_yahoo-finance_19801212-20190531.csv')
    
    # for the sake of simplicity I'm gonna drop Adj Close column
    apple_stock.drop(columns='Adj Close', inplace=True)
    
    if corrected == True:
        apple_stock.loc['1981-08-10'] = (apple_stock.loc['1981-08-07'] + apple_stock.loc['1981-08-11']) / 2
    
    return apple_stock

def get_apple_close_price():
    '''
    Will return a pandas Series with just Close price
    '''
    apple_stock = get_apple_stock()
    return apple_stock['Close']

def get_range(series, start, end=None):
    '''
    Returns a range between start and end
    params:
        series: pandas DataFrame
        start: string - starting date
        end: string - end date
    '''
    if end is not None:
        return series[(series.index >= start) & (series.index <= end)]
    else:
        return series[series.index >= start]

def train_test_split(series, day_split):
    '''
    Train/test split on a specific day
    params:
        series: pandas DataFrame
        day_split: string - when to split
    
    '''
    train = series[series.index <= day_split]
    test = series[series.index > day_split]
    
    return train, test

def plot_field_over_time(series, 
                         y='Close', 
                         xlabel='Year', 
                         ylabel=None, 
                         ylegend=None, 
                         title='', 
                         figsize=(15, 6)):
    '''
    Plots a field (y) over time
    '''
    ax = series.reset_index().plot(x='Date', 
                                   y=y, 
                                   title=title,
                                   figsize=figsize)
    ax.set_xlabel(xlabel)
    
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if ylegend is not None:
        ax.legend([ylegend])

def plot_candlestick(series, xlabel, ylabel, title='', figsize=(15, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    candlestick_ohlc(ax, 
                     zip(mdates.date2num(series.index.to_pydatetime()), 
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

def adf_test(series):
    '''
    Perform Dickey-Fuller test
    see: https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
    '''
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

def plot_series(series, title='', legend=None, figsize=(15, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(series)
    ax.set_title(title)
    if legend is not None:
        ax.legend(legend)

def difference(series):
    '''
    Calculate the n-th order discrete difference
    '''
    return np.diff(series), series[0]

def inverse_difference(series, first_value):
    '''
    Does the inverse of difference
    '''
    return np.hstack((first_value, first_value+np.cumsum(series)))  

def log_transform(series):
    return np.log(series)

def inverse_log_transform(series):
    return np.exp(series)

def rmse(preds, targets):
    '''
    Calculates Root Mean Square Error
    preds: Series of predictions
    targets: Series of real values
    '''
    return np.sqrt(((preds - targets)**2).mean())

def plot_walk_forward_validation(test, predictions, model_name='Model', size=1, steps=1):
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.plot(test[:size])
    plt.plot(predictions)
    ax.set_title('{} - Walk forward validation - {} days, {} days prediction'.format(model_name, 
                                                                                     size, 
                                                                                     steps))
    ax.legend(['Expected', 'Predicted'])

def split_sequence(seq, look_back, n_outputs=1):
    '''
    split a sequence into samples.
    Example:
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        look_back = 3
        n_outputs = 2

            X          y
        -------------------
        [1, 2, 3]   [4,  5]
        [2, 3, 4]   [5,  6]
        [3, 4, 5]   [6,  7]
        [4, 5, 6]   [7,  8]
        [6, 7, 8]   [9, 10]
        
    '''
    X, y = list(), list()
    seq_len = len(seq)
    
    for i in range(seq_len):
        # find the end of this pattern
        target_i = i + look_back
        
        # check if we are beyond the sequence
        if target_i + n_outputs > seq_len: break
        
        # gather input and output parts of the pattern
        seq_x, seq_y = seq[i:target_i], seq[target_i:target_i+n_outputs]
        
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)

def ARIMA_walk_forward_validation(train, test, order, size=1, steps=1, debug=True):
    '''
    Performs a walk-forward validation on ARIMA model
    params:
        train: Series - train set
        test: Series - test set
        order: Tuple - order parameter of ARIMA model
        size: Integer - amount of days we're gonna walk
        steps: Integer - how many days we're gonna forecast
        debug: Bool - prints debug prediction vs expected prices
    '''
    
    history = [x for x in train]
    pred = list()
    limit_range = len(test[:size])

    for t in range(0, limit_range, steps):
        model = ARIMA(history, order=order)
        model_fit = model.fit(disp=0) # trains the model with the new history
        output = model_fit.forecast(steps=steps) # make predictions
        yhat = output[0]
        pred = pred + yhat.tolist()
        obs = test[t:t+steps]
        history = history + obs.values.tolist()
        history = history[len(obs.values):] # shift to forget the oldest prices
        
        if debug == True:
            print('predicted={}, expected={}'.format(yhat, obs.values))
            
    return pred[:limit_range]

def NN_walk_forward_validation(model, 
                               train, test, 
                               size=1, look_back=1, n_outputs=1):
    '''
    Performs a walk-forward validation on a NN model
    params:
        model: NN model
        train: Series - train set
        test: Series - test set
        size: Integer - amount of days we're gonna walk
        look_back: Integer - amount of past days to forecast future ones
        n_outputs: Integer - amount of days predicted (output of predictor)
    '''
    
    past = train.reshape(-1,).copy()
    future = test.reshape(-1,)[:size]
    
    predictions = list()
    limit_range = len(future)

    for t in range(0, limit_range, n_outputs):
        x_input = past[-look_back:] # grab the last look_back days from the past
        x_input = x_input.reshape(1, look_back, 1)
        
        # predict the next n_outputs days
        y_hat = model.predict(x_input)
        predictions.append(y_hat.reshape(n_outputs,))

        # add the next real days to the past
        past = np.concatenate((past, future[t:t+n_outputs]))
        
        if len(future[t:t+n_outputs]) == n_outputs:
            X_batch = x_input
            y_batch = future[t:t+n_outputs].reshape(-1, n_outputs)
        
            # Time to re-train the model with the new non-seen days
            model.train_on_batch(X_batch, y_batch)
            
    return np.array(predictions).reshape(-1,)[:limit_range]

def NN_walk_forward_validation_v2(model,
                                  train, test, 
                                  size=1, 
                                  look_back=1, n_features=1, n_outputs=1):
    
    '''
    Performs a walk-forward validation on a NN model
    when there are multiple features
    '''
    
    past = train.copy()
    future = test[:size]
    
    predictions = list()
    limit_range = len(future)

    for t in range(0, limit_range, n_outputs):
        x_input = past[-look_back:] # grab the last look_back days from the past
        x_input = x_input.reshape(1, look_back, n_features)
        
        # predict the next n_outputs days
        y_hat = model.predict(x_input)
        predictions.append(y_hat.reshape(n_outputs,))

        # add the next real days to the past
        past = np.concatenate((past, future[t:t+n_outputs]))
        
        if len(future[t:t+n_outputs]) == n_outputs:
            X_batch = x_input
            y_batch = future[t:t+n_outputs]
            y_batch = y_batch[:, 3].reshape(-1, n_outputs)
        
            # Time to re-train the model with the new non-seen days
            model.train_on_batch(X_batch, y_batch)
            
    return np.array(predictions).reshape(-1,)[:limit_range]

def plot_columns(series, columns, fromTo=None, figsize=(15, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    if fromTo is not None:
        series = get_range(series, fromTo[0], fromTo[1])
        
    for column in columns:
        series.reset_index().plot(ax=ax, x='Date', y=column)

def freeze_layers(model, freeze=True):
    for layer in model.layers:
        layer.trainable = freeze

def calculate_forecast_error(preds, targets):
    preds, targets = np.array(preds), np.array(targets)
    return targets - preds

def plot_residual_forecast_error(preds, targets, figsize=(15, 6)):
    forecast_errors = calculate_forecast_error(preds, targets)
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(forecast_errors)
    plt.axhline(y=0, color='grey', linestyle='--', )
    ax.set_title('Residual Forecast Error')

def mape(preds, targets): 
    return np.mean(np.abs((targets - preds)/targets)) * 100
    
def print_performance_metrics(preds, targets, total_days=21, steps=1, model_name=''):
    '''
    Prints a report with different metrics:
    Inspired by 
        https://machinelearningmastery.com/time-series-forecasting-performance-measures-with-python/
    '''
    
    preds, targets = np.array(preds), np.array(targets)
    forecast_errors = calculate_forecast_error(preds, targets)
    
    print('%s[%d days, %d days forecast]:\n' % (model_name, total_days, steps))
    print('Forecast Bias: %.3f' % (np.sum(forecast_errors)*1.0/len(targets)))
    print('MAE: %.3f' % (np.mean(np.abs(forecast_errors))))
    print('MSE: %.3f' % (np.mean(forecast_errors**2)))
    print('RMSE: %.3f' % (rmse(preds, targets)))
    print('MAPE: %.3f' % (mape(preds, targets)))

def descale_with_features(predictions, 
                          test, 
                          n_features, 
                          scaler=None,
                          transformer=None,
                          pos_to_fillin=3):
    '''
    In order to be able to de-scale the price, we need to
    create a table with n_features columns and place the 
    predicted Close price in position 3
    '''
    
    ret_preds = np.zeros((predictions.shape[0], n_features))
    ret_preds[:, pos_to_fillin] = predictions
    ret_test = test

    if scaler is not None:
        ret_preds = scaler.inverse_transform(ret_preds)
        ret_test = scaler.inverse_transform(ret_test)
    
    if transformer is not None:
        ret_preds = transformer.inverse_transform(ret_preds)
        ret_test = transformer.inverse_transform(ret_test)
    
    return ret_preds[:, pos_to_fillin], ret_test