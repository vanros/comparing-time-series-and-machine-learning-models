
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
from statsmodels.tsa.stattools import adfuller
from matplotlib.dates import DateFormatter, MonthLocator, DayLocator
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('C://Users//vanessa_rodrigues//Documents//Pesquisa-mestrado/dados-FEAAC//2018-2019-2020//consumption-feaac-2018-2019-2020.csv')


data = data.rename(columns={'Data': 'date', 'Consumo (kWh)': 'data'})



data['data'] = data['data'].replace(',','.', regex=True).astype(float)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M:%S',
                               errors='coerce')
data = data.set_index('date')
data = data.resample('D').sum()






data_monthly = data

data_monthly = data_monthly.resample('M').sum()

data_monthly.plot(figsize=(15, 6))
plt.show()




plt.figure(figsize=(20,30))
plt.plot(data_monthly.index.values, data_monthly['data'])
plt.ylabel('Consumo (kWh)')
plt.tick_params(labelsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xticks(rotation=45)
dtFmt = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_locator(DayLocator(interval=30)) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) # apply the format to the desired axis
plt.show()



from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

data_scaled = min_max_scaler.fit_transform(data_monthly)
data_scaled = pd.DataFrame(data_scaled, index = data_monthly.index)

train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size

tr_start,tr_end = '2018-02-25','2019-09-29'
te_start,te_end = '2019-10-06','2020-03-01'


train, test = data_scaled.iloc[0:train_size,:], data_scaled.iloc[train_size:len(data_scaled),:]


#we would do first-order differencing for the trend and re-run the ADF test to check for stationarity.


ts_t_adj = data_monthly - data_monthly.shift(1)
ts_t_adj = ts_t_adj.dropna()
ts_t_adj.plot()


ts_s_adj = ts_t_adj

# seasonal differencing
ts_s_adj = ts_t_adj - ts_t_adj.shift(6)
ts_s_adj = ts_s_adj.dropna()
ts_s_adj.plot()



from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(train)


plot_acf(train, lags=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plot_pacf(train, lags=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


results_list = []
#set parameter range
p = range(0,3)
q = range(1,3)
d = range(1,2)
s = range(4,7)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(p, d, q, s))
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))



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









from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(data_monthly, lags=12)
matplotlib.pyplot.show()
plot_pacf(data_monthly, lags=12)
matplotlib.pyplot.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(data_monthly, model='additive')
fig = decomposition.plot()
plt.show()




#ADF: The intuition behind a unit root test is that it determines how strongly a time series is defined by a trend.
#p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
#p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
#OSCB: if the value is less than 0.64, the series is stationary


def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number  of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

print(adf_test(data_monthly))

import pmdarima

results = pmdarima.arima.OCSBTest(m=12).estimate_seasonal_differencing_term(data_monthly)
print(results)
#we would do first-order differencing for the trend and re-run the ADF test to check for stationarity.



#We will then pick the model with the least AIC

p = range(0, 3)
d = range(1,2)
q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data_monthly['data'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


#SARIMA(0, 1, 0)x(0, 1, 1, 12)12
from sklearn import preprocessing
import time
from sklearn.metrics import r2_score,mean_squared_error
from statistics import mean



min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

data_scaled = min_max_scaler.fit_transform(data_monthly)
data_scaled = pd.DataFrame(data_scaled, index = data_monthly.index)

train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size

tr_start,tr_end = '2018-02-28','2019-09-30'
te_start,te_end = '2019-11-31','2020-02-29'

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
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1,6)).fit()
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


plt.plot(test_scaled, label='Test ')
plt.plot(pred_scaled, label='Predições')
plt.legend(framealpha=1, frameon=True);
plt.xlabel('Data')
plt.ylabel('Consumption (kWh)')
dtFmt = mdates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.show()






