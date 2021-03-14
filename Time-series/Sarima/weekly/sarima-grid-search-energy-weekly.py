
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
from sklearn.model_selection import train_test_split
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

data.to_csv('energy-daily-data.csv')

plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
plt.plot(data['date'], data['data'])
plt.title('Energy Consumption')
plt.ylabel('Energy Consumption')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


plot_pacf(data['data']);
plot_acf(data['data']);


#p-value is less than the critical value of 0.05

#Now, let’s take the log difference in an effort to make it stationary:
#data['data'] = np.log(data['data'])
#data['data'] = data['data'].diff()
#data = data.drop(data.index[0])

#data-weekly

data_weekly = data

data_weekly = data_weekly['2018-02-25':]

data_weekly = data_weekly.resample('W').sum()




data_plot = data_weekly.loc['2018-02-17':'2018-12-30']

plt.figure(figsize=(20,30))
plt.plot(data_weekly.index.values, data_weekly['data'])
plt.ylabel('Consumo (kWh)')
plt.tick_params(labelsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xticks(rotation=45)
dtFmt = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_locator(DayLocator(interval = 7)) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) # apply the format to the desired axis
plt.show()



train_size = int(len(data_weekly) * 0.8)
test_size = len(data_weekly) - train_size


train, test = data_weekly.iloc[0:train_size,:], data_weekly.iloc[train_size:len(data_weekly),:]

#Dickey-Fuller test



#we would do first-order differencing for the trend and re-run the ADF test to check for stationarity.


ts_t_adj = data_weekly - data_weekly.shift(1)
ts_t_adj = ts_t_adj.dropna()
ts_t_adj.plot()


ts_s_adj = ts_t_adj

# seasonal differencing
ts_s_adj = ts_t_adj - ts_t_adj.shift(22)
ts_s_adj = ts_s_adj.dropna()
ts_s_adj.plot()

ts_s_adj_plot = ts_s_adj.loc['2018-02-17':'2018-12-30']

plt.figure(figsize=(20,30))
plt.plot(ts_s_adj_plot.index.values, ts_s_adj_plot['data'])
plt.ylabel('Consumo (kWh)')
plt.tick_params(labelsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xticks(rotation=45)
dtFmt = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_locator(DayLocator(interval = 7)) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) # apply the format to the desired axis
plt.show()




train_size = int(len(ts_s_adj) * 0.8)
test_size = len(ts_s_adj) - train_size


train, test = ts_s_adj.iloc[0:train_size,:], ts_s_adj.iloc[train_size:len(ts_s_adj),:]



#Dickey-Fuller test

from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(train)



#Dickey-Fuller test

from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(train)



plot_acf(train, lags=50)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plot_pacf(train, lags=50)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


results_list = []
#set parameter range
p = range(0,3)
q = range(0,3)
d = range(0,2)
s = range(20,23)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(p, d, q, s))

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


from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

data_scaled = min_max_scaler.fit_transform(data_weekly)
data_scaled = pd.DataFrame(data_scaled, index = data_weekly.index)

train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size

tr_start,tr_end = '2018-02-25','2019-09-29'
te_start,te_end = '2019-10-06','2020-03-01'

train, test = data_scaled.iloc[0:train_size,:], data_scaled.iloc[train_size:len(data_scaled),:]


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
                                order=(1, 1, 1),
                                seasonal_order=(0, 0, 0, 21)).fit()
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


















#we would do first-order differencing for the trend and re-run the ADF test to check for stationarity.



ts_t_adj = data_weekly - data_weekly.shift(1)
ts_t_adj = ts_t_adj.dropna()
ts_t_adj.plot()

print(adf_test(ts_t_adj))

ts_s_adj = ts_t_adj

# seasonal differencing
ts_s_adj = ts_t_adj - ts_t_adj.shift(6)
ts_s_adj = ts_s_adj.dropna()
ts_s_adj.plot()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(ts_s_adj)
matplotlib.pyplot.show()
plot_pacf(ts_s_adj)
matplotlib.pyplot.show()   


from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

data_scaled = min_max_scaler.fit_transform(data_weekly)
data_scaled = pd.DataFrame(data_scaled, index = data_weekly.index)

train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size

tr_start,tr_end = '2018-02-25','2019-09-29'
te_start,te_end = '2019-10-06','2020-03-01'


train, test = data_scaled.iloc[0:train_size,:], data_scaled.iloc[train_size:len(data_scaled),:]





mse_list = []
rmse_list = []

from sklearn.metrics import mean_squared_error
from statistics import mean 


for i in range(30):
    mod = sm.tsa.statespace.SARIMAX(train,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12)).fit()
    pred = mod.predict(tr_end,te_end)[1:]
    mse = mean_squared_error(test,pred)
    rmse = np.sqrt(mse)
    mse_list.append(mse)
    rmse_list.append(rmse)

print('SARIMA model MSE:{}'.format(mean(mse_list)))
print('SARIMA model RMSE:{}'.format(mean(rmse_list)))


test_scaled = min_max_scaler.inverse_transform(test)
pred = pred.to_frame()
pred_scaled = min_max_scaler.inverse_transform(pred)

test_scaled = pd.DataFrame(test_scaled, index = test.index)
pred_scaled = pd.DataFrame(pred_scaled, index = pred.index)


plt.plot(test_scaled, label='Test ')
plt.plot(pred_scaled, label='Predictions')
plt.legend(framealpha=1, frameon=True);
plt.tick_params(labelsize=10)
plt.yticks(fontsize=10)
plt.xticks(rotation=40)
plt.ylabel('Consumption (kWh)')
dtFmt = mdates.DateFormatter('%d-%m-%Y')
#plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) 
plt.show()



#GRID SEARCH
p = range(0,3)
q = range(0,3)
d = range(0,2)
s = range(20,22)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(p, d, q, s))
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
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







#walk forward validation 




# data = entire dataset
# n_test = point where data is split into training and test sets
def walk_forward_validation(data, n_test):
    predictions = np.array([])
    rmse_list = []
    train, test = data[:n_test], data[n_test:]
    day_list = [7,14,21,28] # weeks 1,2,3,4
    for i in day_list:
        # Fit model to training data
        model = sm.tsa.statespace.SARIMAX(train, 
                                          holiday_ex_var,
                                          order=(1,1,2), 
                                          seasonal_order(1,1,2,7)).fit(max_iter = 50,
                                              method = 'powell')
        
        # Forecast daily loads for week i
        forecast = model.get_forecast(steps = 7)
        predictions = np.concatenate(predictions, forecast, 
                                     axis=None)
        # Calculate MAPE and add to mape_list
        rmse_score = np.sqrt(mean_squared_error(test,pred))
        rmse_list.append(rmse_score)
        # Add week i to training data for next loop
        train = np.concatenate((train, test[j:i]), axis=None)
return predictions, mape_list


























