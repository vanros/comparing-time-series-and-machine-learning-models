
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MonthLocator, DayLocator
import pandas as pd
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import mean_absolute_error



data = pd.read_csv('C://Users//vanessa_rodrigues//Documents//Pesquisa-mestrado/dados-FEAAC//2018-2019-2020//consumption-feaac-2018-2019-2020.csv')


data = data.rename(columns={'Data': 'date', 'Consumo (kWh)': 'data'})



data['data'] = data['data'].replace(',','.', regex=True).astype(float)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M:%S',
                               errors='coerce')
data = data.set_index('date')
data = data.resample('D').sum()


plt.figure(figsize=[15, 70]); # Set dimensions for figure
fig,ax1 = plt.subplots()
plt.plot(data.index.values, data['data'])
#plt.title('Energy Consumption')
plt.ylabel('Consumo  (kWh)')
plt.xlabel('Data')
plt.xlim([1, 3])
plt.tick_params(labelsize=8)
plt.xticks(fontsize=10)
plt.xticks(rotation=20)
monthyearFmt = mdates.DateFormatter('%Y %Y')
ax1.xaxis.set_major_formatter(monthyearFmt)

plt.grid(True)
plt.show()




plt.figure(figsize=(20,30))
plt.plot(data.index.values, data['data'])
plt.ylabel('Consumo (kWh)', fontsize=25)
#plt.xlabel('Date')
plt.tick_params(labelsize=12)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(rotation=35)
dtFmt = mdates.DateFormatter('%m-%Y')
plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) # apply the format to the desired axis
plt.show()



data_plot = data.loc['2019-09-01':'2019-09-30']

plt.figure(figsize=(20,30))
plt.plot(data_plot.index.values, data_plot['data'])
plt.ylabel('Consumo (kWh)')
plt.tick_params(labelsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xticks(rotation=45)
dtFmt = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_locator(DayLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) # apply the format to the desired axis
plt.show()



#we would do first-order differencing for the trend and re-run the ADF test to check for stationarity.


ts_t_adj = data - data.shift(1)
ts_t_adj = ts_t_adj.dropna()
ts_t_adj.plot()


ts_s_adj = ts_t_adj

# seasonal differencing
ts_s_adj = ts_t_adj - ts_t_adj.shift(7)
ts_s_adj = ts_s_adj.dropna()
ts_s_adj.plot()

ts_s_adj_plot = ts_s_adj.loc['2019-09-01':'2019-09-30']

plt.figure(figsize=(20,30))
plt.plot(ts_s_adj_plot.index.values, ts_s_adj_plot['data'])
plt.ylabel('Consumo (kWh)')
plt.tick_params(labelsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xticks(rotation=45)
dtFmt = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_locator(DayLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) # apply the format to the desired axis
plt.show()





train_size = int(len(data) * 0.8)
test_size = len(data) - train_size


train, test = data.iloc[0:train_size,:], data.iloc[train_size:len(data),:]



#Dickey-Fuller test

from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(train)


#SARIMA(p,d,q)x(P,D,Q,s)

import matplotlib.pyplot as plt
import statsmodels.api as sm

matplotlib.rcParams.update({'font.size': 10})
figsize = (4,3) # same ratio, bigger text
fig,ax = plt.subplots(2, 1, figsize=figsize)
sm.graphics.tsa.plot_acf(train, lags=50, ax=ax[0])
sm.graphics.tsa.plot_pacf(train, lags=50, ax=ax[1])
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


fig = plt.subplots(2, 1)

plt.figure(figsize=(30,15))
plot_acf(train, lags=50)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.figure(figsize=(30,15))
plot_pacf(train, lags=50)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)



sm.graphics.tsa.plot_acf(train, lags=40, ax=ax[0])
plt.xlabel('lag')
plt.ylabel('correlação')
sm.graphics.tsa.plot_pacf(train, lags=40, ax=ax[1])
plt.xlabel('lag')
plt.ylabel('correlação')
plt.show()








results_list = []
#set parameter range
p = range(0,3)
q = range(0,3)
d = range(0,2)
s = range(7,8)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(p, d, q, s))
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train,
                                            order=param,
                                            seasonal_order=param_seasonal)
            results = mod.fit(max_iter = 50, method = 'powell')
            print('SARIMA{}x{}7- AIC:{}'.format(param, param_seasonal, results.aic))
            results_list.append(results.aic)
        except:
            continue

print('SARIMA BEST', min(results_list))

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

data_scaled = min_max_scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, index = data.index)

train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size

tr_start,tr_end = '2018-02-21','2019-10-04'
te_start,te_end = '2019-10-05','2020-02-29'


train, test = data_scaled.iloc[0:train_size,:], data_scaled.iloc[train_size:len(data_scaled),:]

#SARIMA(2, 1, 2)x(0, 1, 2, 12)12
mse_list = []
rmse_list = []

from sklearn.metrics import mean_squared_error
from statistics import mean 

rmse_list_testError = []
r2_list_testError = []
mae_list = []


rmse_list_trainError = []
r2_list_trainError = []
training_time_list =[]

predict_time_list = []


for i in range(30):
    
    t_start_training = time.time()
    mod = sm.tsa.statespace.SARIMAX(train,
                                order=(2, 1, 2),
                                seasonal_order=(2, 1, 2, 7)).fit()
    t_training = round( time.time()-t_start_training, 3) # the time would be round to 3 decimal in seconds
    training_time_list.append(t_training)
    
    
    t_start_predict =  time.time()
    pred = mod.predict(tr_end,te_end)[1:]
    t_predict = round( time.time()-t_start_predict, 3)
    predict_time_list.append(t_predict)
    

    rmse_list_testError.append(np.sqrt(mean_squared_error(test,pred)))
    r2_list_testError.append(r2_score(test,pred))
    mae_list.append(mean_absolute_error(test, pred))
    
    
    

print('RMSE_TestError:', mean(rmse_list_testError))
print('MAE:', mean(mae_list))

print('R-squared_TestError:', mean(r2_list_testError))

print('Training  Time (s)', mean(training_time_list))
print('Predict  Time (s)', mean(predict_time_list))



test_scaled = min_max_scaler.inverse_transform(test)
pred = pred.to_frame()
pred_scaled = min_max_scaler.inverse_transform(pred)

test_scaled = pd.DataFrame(test_scaled, index = test.index)
pred_scaled = pd.DataFrame(pred_scaled, index = pred.index)


plt.plot(test_scaled, label='Teste ')
plt.plot(pred_scaled, label='Predições')
plt.legend(framealpha=1, frameon=True);
plt.xlabel('Data')
plt.ylabel('Consumo (kWh)')
dtFmt = mdates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.show()

plt.plot(min_max_scaler.inverse_transform(data_scaled), label='Dados')
plt.plot(test_scaled, label='Teste')
plt.plot(pred_scaled, label='Predições')
plt.legend(framealpha=1, frameon=True);
plt.xlabel('Date')
plt.ylabel('Consumption (kWh)')
plt.show()

plt.plot(test_scaled, label='Teste ')
plt.plot(pred_scaled, label='Predições')
plt.ylabel('Consumo (kWh)')
plt.xlabel('Date')
plt.tick_params(labelsize=10)
plt.yticks(fontsize=10)
plt.xticks(rotation=40)
dtFmt = mdates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) # apply the format to the desired axis
plt.show()


# Set variables to populate
lowest_aic = None
lowest_parm = None
lowest_param_seasonal = None
# GridSearch the hyperparameters of p, d, q and P, D, Q, m
# GridSearch the hyperparameters of p, d, q and P, D, Q, m
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mdl = sm.tsa.statespace.SARIMAX(train, order=param, seasonal_order=param_seasonal, enforce_stationarity=True, enforce_invertibility=True)
            results = mdl.fit()
            
            # Store results
            current_aic = results.aic
            # Set baseline for aic
            if (lowest_aic == None):
                lowest_aic = results.aic
            # Compare results
            if (current_aic <= lowest_aic):
                lowest_aic = current_aic
                lowest_parm = param
                lowest_param_seasonal = param_seasonal
            print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
            
print('The best model is: SARIMA{}x{} - AIC:{}'.format(lowest_parm, lowest_param_seasonal, lowest_aic))


def walk_forward_validation(data, n_test):
    predictions = []
    rmse_list = []
    train, test = data[:n_test], data[n_test:]

    day_list = [7,14,21,28] # weeks 1,2,3,4
    for i in day_list:
        # Fit model to training data
        model = sm.tsa.statespace.SARIMAX(train,
                                          order=(2, 1, 2),
                                          seasonal_order=(2, 1, 1, 7)).fit(max_iter = 50,
                                              method = 'powell')
        
        # Forecast daily loads for week i
        forecast = model.forecast(steps = 7)
        forecast = np.array(forecast, dtype=float)
        predictions.append(forecast)
        # Calculate MAPE and add to mape_list
        j = i-7
        rmse_score = np.sqrt(mean_squared_error(test,pred))
        rmse_list.append(rmse_score)
        # Add week i to training data for next loop
        train = np.concatenate((train, test[j:i]), axis=0)
        return predictions, rmse_list
    
train_size = int(len(ts_s_adj) * 0.8)
test_size = len(ts_s_adj) - train_size


pred, rmse_list = walk_forward_validation(ts_s_adj, train_size)


data_scaled = ts_s_adj



from statsmodels.tsa.stattools import adfuller
def test_stationarity(data):
    p_val=adfuller(data)[1]
    if p_val >= 0.05:
        print("Time series data is not stationary. Adfuller test pvalue={}".format(p_val))
    else:
        print("Time series data is stationary. Adfuller test pvalue={}".format(p_val))
test_stationarity(train) 









import numpy as np
from math import sqrt
from multiprocessing import cpu_count
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
 
# Direct method for SARIMA forecast
# A new model for every prediction
def sarima_forecast(history):
    #order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=(2, 1, 2), seasonal_order=(2, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]

def mean_absolute_percentage_error(y_true, y_pred):
    '''Take in true and predicted values and calculate the MAPE score.'''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    pred = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history)
        # store forecast in list of predictions
        pred.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = mean_absolute_percentage_error(test, pred)
    return pred, error

# score a model, return None on failure
def score_model(data, n_test, debug=False):
    result = None
    # convert config to a key
    #key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                predictions, result = walk_forward_validation(data, n_test)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (result))
    return predictions, result


order=(2, 1, 2)
seasonal_order=(2, 1, 1, 7)
 
pred, results = score_model(ts_s_adj, test_size, debug=False)


train_scaled = min_max_scaler.inverse_transform(train)
train_scaled = train_scaled.to_frame()
pred_scaled = min_max_scaler.inverse_transform(pred)


from statsmodels.tsa.stattools import adfuller
def test_stationarity(data):
    p_val=adfuller(data)[1]
    if p_val >= 0.05:
        print("Time series data is not stationary. Adfuller test pvalue={}".format(p_val))
    else:
        print("Time series data is stationary. Adfuller test pvalue={}".format(p_val))
test_stationarity(train) 