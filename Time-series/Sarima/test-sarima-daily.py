# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 19:38:32 2021

@author: vanessa_rodrigues
"""
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



data = pd.read_csv('C://Users//vanessa_rodrigues//Documents//Pesquisa-mestrado/dados-FEAAC//2018-2019-2020//consumption-feaac-2018-2019-2020.csv')


data = data.rename(columns={'Data': 'date', 'Consumo (kWh)': 'data'})



data['data'] = data['data'].replace(',','.', regex=True).astype(float)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M:%S',
                               errors='coerce')
data = data.set_index('date')
data = data.resample('D').sum()

data.plot(figsize=(12,12))


from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

rcParams['figure.figsize']=18,8
decompostion=sm.tsa.seasonal_decompose(data)
decompostion.plot()
plt.tick_params(labelsize=10)
plt.yticks(fontsize=10)
plt.show()

count = (data < 0).sum().sum()
data.isnull().values.any()




roll_mean=data.rolling(12).mean()
roll_std=data.rolling(12).std()
plt.plot(data, color='black',label='Original')
plt.plot(roll_mean,color="blue",label='Rolling Mean')
plt.plot(roll_std,color='green',label='Rolling Standard Deviation')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)

X=data.values
split=round(len(X)/2)
X1=X[0:split]
X2=X[split:]
mean1,mean2=X1.mean(),X2.mean()
var1,var2=X1.var(),X2.var()


print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

# No constant mean
# No constant variance
# the Dataset is non-stationary


#Using Augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
def dickey_test(data):
    X_dickey=data.values
    result = adfuller(X_dickey)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

dickey_test(data)

from statsmodels.tsa.stattools import kpss
def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

kpss_test(data.values)


import statsmodels.api as sm
fig, ax = plt.subplots(2,1)
fig = sm.graphics.tsa.plot_acf(data,  ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(data, ax=ax[1], method='ywm' )
plt.show()

moving_average=data.rolling(window=12).mean()
plt.plot(data)
plt.plot(moving_average,color='red')

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size

tr_start,tr_end = '2018-03-04','2019-10-06'
te_start,te_end = '2019-10-13','2020-03-01'

train, test = data.iloc[0:train_size,:], data.iloc[train_size:len(data),:]


train2=data.copy()
train = np.log(train)

model = sm.tsa.statespace.SARIMAX(train,
                                order=(2, 1, 2),
                                seasonal_order=(2, 1, 1, 7)).fit(max_iter = 50, method = 'powell')


from math import exp
def unlog_pred(log_pred):
    
    pred = [exp(i) for i in log_pred.values]
    pred = pd.Series(pred, index = log_pred.index)
    
    return pred
pred_forecast_log = model.get_forecast(steps = 148)
# pred_uc_ci = pred_forecast_log.conf_int(alpha=0.05)
pred_forecast = model.get_forecast(steps = 148).predicted_mean

test['Prediction']=pred_forecast


true_value=test['data'].values
forcast=test['Prediction'].values
mse_score = ((forcast- true_value) ** 2).mean()
print('MSE of our forecasts is {}'.format(round(mse_score, 3)))
rmse = np.sqrt(mse_score)
print("RMSE-Value" ,rmse)

mape = (abs(forcast-true_value)/true_value)*100
mape.mean()


ax=data.plot()
test['Prediction'].plot(ax=ax,label='Predicted')






from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")
entire_dataset= np.log(data.values)
n_test=148
def walk_forward_validation(data, n_test):  
    predictions =[]
    log_predictions=[]
    mape_list = []
    test_ind=len(data)-n_test
    train, test = data[:test_ind], data[test_ind:]  
    train,test=train_test_split(data,test_size=n_test)
    history=[x for x in data]
        
    for i in range(len(test)):
        
        
        # Fit model to training data
        model=sm.tsa.statespace.SARIMAX(history,order=(2,1,2),seasonal_order=(2,1,2,7)).fit(max_iter=50,method="powell")

        
        # Forecast daily loads for week i
        forecast = model.predict(len(history),len(history))
        output=forecast[0]
        predictions.append(forecast)      
        #Calculate MAPE and add to mape_list
        obs=test[i]
        print('predicted=%f, expected=%f' % (output, obs))
        history.append(obs)
        
#     #Calculate MAPE and add to mape_list   
# #     pred_forecast = unlog_pred(log_predictions.predicted_mean)
# #     predictions.append(pred_forecast)
    
            

   
# #     true_value=np.exp(test.values)
# #     forcast=pred_forecast.values
#     print('predicted=%f, expected=%f' % (pred, true_value))
        
#     mape = (abs(forcast-true_value)/true_value)*100
#     mape_list.append(mape)
        
    
    error = mean_squared_error(test, predictions)
    rmse = mean_squared_error(test, predictions)**0.5
        
    return predictions,error,rmse



preddictions,error,rmse=walk_forward_validation(entire_dataset,n_test=n_test)

print('Test RMSE: %.3f' % rmse)



test_ind=len(entire_dataset)-n_test
# train, test = data[:test_ind], entire_dataset[test_ind:]  
train,test=train_test_split(entire_dataset,test_size=n_test)



predictions = pd.Series(preddictions, copy=True)
predictions=predictions.values.reshape(predictions.shape[0],-1)
predictions.shape

plt.plot(preddictions)
plt.plot(test)

true_value=test
forcast=predictions

mape = (abs(forcast-true_value)/true_value)*100
mape.mean()
















