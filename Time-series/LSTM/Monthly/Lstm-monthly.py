# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:15:28 2021

@author: vanessa_rodrigues
"""

from numpy import array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

np.random.seed(7)

data = pd.read_csv('C://Users//vanessa_rodrigues//Documents//Pesquisa-mestrado/dados-FEAAC//2018-2019-2020//consumption-feaac-2018-2019-2020.csv')


data = data.rename(columns={'Data': 'date', 'Consumo (kWh)': 'data'})



data['data'] = data['data'].replace(',','.', regex=True).astype(float)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M:%S',
                               errors='coerce')
data = data.set_index('date')
data = data.resample('D').sum()

data_monthly = data
data_monthly = data_monthly.resample('M').sum()


#normalize dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data_monthly_scaled = scaler.fit_transform(data_monthly)
data_monthly_scaled = pd.DataFrame(data_monthly_scaled, index = data_monthly.index)


# split into train and test sets
train_size = int(len(data_monthly_scaled) * 0.8)
test_size = len(data_monthly_scaled) - train_size
train, test = data_monthly_scaled.iloc[0:train_size,:], data_monthly_scaled.iloc[train_size:len(data_monthly_scaled),:]


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train.values, look_back)
testX, testY = create_dataset(test.values, look_back)


#[samples, time steps, features].
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# create and fit the LSTM network
batch_size = 1

mse_list = []
rmse_list = []

from sklearn.metrics import mean_squared_error
from statistics import mean 



# create and fit the LSTM network
rmse_list_testError = []
r2_list_testError = []
mae_list = []


rmse_list_trainError = []
r2_list_trainError = []
training_time_list =[]

predict_time_list = []

from sklearn.metrics import r2_score,mean_squared_error
from statistics import mean 
from sklearn.preprocessing import MinMaxScaler
import time
# create and fit the LSTM network
for i in range(30):
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    t_start_training = time.time()
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    t_training = round( time.time()-t_start_training, 3) # the time would be round to 3 decimal in seconds
    training_time_list.append(t_training)
    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size)
    model.reset_states()
    
    t_start_predict =  time.time()
    testPredict = model.predict(testX, batch_size=batch_size)
    t_predict = round( time.time()-t_start_predict, 3)
    predict_time_list.append(t_predict)
    
    rmse_list_testError.append(np.sqrt(mean_squared_error(testY, testPredict)))
    r2_list_testError.append(r2_score(testY, testPredict))
    mae_list.append(mean_absolute_error(testY, testPredict))
    
print('RMSE_TestError:', mean(rmse_list_testError))
print('MAE:', mean(mae_list))
print('R-squared_TestError:', mean(r2_list_testError))

print('Training  Time (s)', mean(training_time_list))
print('Predict  Time (s)', mean(predict_time_list))



# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# shift train predictions for plotting
trainPredictPlot = np.empty_like(data_monthly)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data_monthly)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data_monthly)-1, :] = testPredict

test.drop(test.head(1).index, inplace=True)
test.drop(test.tail(1).index, inplace=True)
testPlot = np.empty_like(data_monthly)
testPlot[:, :] = np.nan
testPlot[len(trainPredict)+(look_back*2)+1:len(data_monthly)-1, :] = test
# plot baseline and predictions
plt.plot(data_monthly, label='Data')
#plt.plot(trainPredictPlot, label='Train Predictions')
plt.plot(testPredictPlot, label='Test Predictions')
plt.legend(framealpha=1, frameon=True);
plt.xlabel('Date')
plt.ylabel('Consumption (kWh)')
plt.show()

plt.plot(scaler.inverse_transform(test), label='Test ')
plt.plot(testPredict, label='Predictions')
plt.legend(framealpha=1, frameon=True);
plt.xlabel('Date')
plt.ylabel('Consumption (kWh)')
plt.show()




testPredictPlot = pd.DataFrame(testPredictPlot, index = data_monthly_scaled.index)
data_monthly_scaled =scaler.inverse_transform(data_monthly_scaled)
data_monthly_scaled = pd.DataFrame(data_monthly_scaled, index = data_monthly_scaled.index)

testPlot = pd.DataFrame(testPlot, index = data_monthly_scaled.index)


testPlot =scaler.inverse_transform(testPlot)
testPlot = pd.DataFrame(testPlot, index = data_monthly_scaled.index)

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MonthLocator, DayLocator


plt.plot(testPlot, label='Teste')
#plt.plot(trainPredictPlot)
plt.plot(testPredictPlot, label='Predições')
plt.legend(framealpha=1, frameon=True);
plt.xlabel('Data')
plt.ylabel('Consumon (kWh)')
plt.show()
dtFmt = mdates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.show()
