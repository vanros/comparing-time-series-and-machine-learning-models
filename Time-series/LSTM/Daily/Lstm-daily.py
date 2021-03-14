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
from tensorflow.python.keras.layers import LSTM, Bidirectional, Conv1D, AveragePooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import mean_absolute_error


np.random.seed(7)

data = pd.read_csv('C://Users//vanessa_rodrigues//Documents//Pesquisa-mestrado/dados-FEAAC//2018-2019-2020//consumption-feaac-2018-2019-2020.csv')


data = data.rename(columns={'Data': 'date', 'Consumo (kWh)': 'data'})



data['data'] = data['data'].replace(',','.', regex=True).astype(float)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M:%S',
                               errors='coerce')
data = data.set_index('date')
data = data.resample('D').sum()


data_weekly = data

data_weekly = data_weekly['2018-02-25':]

data_weekly = data_weekly.resample('W').sum()


#normalize dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, index = data.index)


# split into train and test sets
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, index = data.index)

train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size

train, test = data_scaled.iloc[0:train_size,:], data_scaled.iloc[train_size:len(data_scaled),:]


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

rmse_list_testError = []
r2_list_testError = []
mae_list = []


rmse_list_trainError = []
r2_list_trainError = []
training_time_list =[]

predict_time_list = []

from sklearn.metrics import mean_squared_error
from statistics import mean 


for i in range(30):
    model = Sequential()
    model.add(LSTM(50, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(10, batch_input_shape=(batch_size, look_back, 1), stateful=True))
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
    
    
for i in range(1):
    model = Sequential()
    model.add(LSTM(256, input_shape=(batch_size, look_back)))
    #model.add(LSTM(50))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    t_start_training = time.time()
    model.fit(trainX, trainY, epochs=100, batch_size=8, verbose=2)
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
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict

test.drop(test.head(1).index, inplace=True)
test.drop(test.tail(1).index, inplace=True)
testPlot = np.empty_like(data)
testPlot[:, :] = np.nan
testPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = test
# plot baseline and predictions
plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

testPredictPlot = pd.DataFrame(testPredictPlot, index = data_scaled.index)
testPlot = pd.DataFrame(testPlot, index = data_scaled.index)


testPlot =scaler.inverse_transform(testPlot)
testPlot = pd.DataFrame(testPlot, index = data.index)

plt.plot(testPlot, label='Teste')
#plt.plot(trainPredictPlot)
plt.plot(testPredictPlot, label='Predições')
plt.legend(framealpha=1, frameon=True);
plt.xlabel('Data')
plt.ylabel('Consumon (kWh)')
plt.show()
dtFmt = mdates.DateFormatter('%d-%m-%Y')
#plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.show()


plt.plot(testY, label='Teste')
plt.plot(testPredictPlot, label='Predições')
plt.legend(framealpha=1, frameon=True);
plt.xlabel('Date')
plt.ylabel('Consumption (kWh)')
plt.show()




test = scaler.inverse_transform(test)
plt.plot(test, label='Teste ')
plt.plot(testPredict, label='Predições')
plt.legend(framealpha=1, frameon=True);
plt.xlabel('Data')
plt.ylabel('Consumo (kWh)')
dtFmt = mdates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.show()