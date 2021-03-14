# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:40:18 2021

@author: vanessa_rodrigues
"""

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


data = pd.read_csv('series-todataframe-weekly.csv')

data = data.drop('Unnamed: 0', 1)

data = data.rename(columns={'var1(t-1)': 'X', 'var1(t)': 'y'})


X = data.X.values.reshape(-1,1)
y = data.y.values.reshape(-1,1)

mse_list_testError = []
rmse_list_testError = []
mae_list = []


scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.20, random_state=42)


rmse_list_testError = []
r2_list_testError = []


rmse_list_trainError = []
r2_list_trainError = []
training_time_list =[]

predict_time_list = []

for i in range(30):
    svr = SVR(C = 1.5, epsilon = 0.1, gamma = 1e-07, kernel = 'linear')
    
    t_start_training = time.time()
    svr.fit(X_train, y_train)
    t_training = round( time.time()-t_start_training, 3) # the time would be round to 3 decimal in seconds
    training_time_list.append(t_training)
    
    t_start_predict =  time.time()
    y_pred_test = svr.predict(X_test)
    t_predict = round( time.time()-t_start_predict, 3)
    predict_time_list.append(t_predict)
    
    y_pred_train = svr.predict(X_train)
    

    rmse_list_testError.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2_list_testError.append(r2_score(y_test, y_pred_test))
    mae_list.append(mean_absolute_error(y_test, y_pred_test))
    
    rmse_list_trainError.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    r2_list_trainError.append(r2_score(y_train, y_pred_train))
    
    

print('RMSE_TestError:', mean(rmse_list_testError))
print('RMSE_TrainError:', mean(rmse_list_trainError))
print('MAE:', mean(mae_list))


print('R-squared_TestError:', mean(r2_list_testError))
print('R-squared_TrainError:', mean(r2_list_trainError))

print('Training  Time (s)', mean(training_time_list))
print('Predict  Time (s)', mean(predict_time_list))

y_pred_test = y_pred_test.reshape(21, 1)

plt.plot(scaler.inverse_transform(y_test), label='Teste ')
plt.plot(scaler.inverse_transform(y_pred_test), label='Predições')
plt.legend(framealpha=1, frameon=True);
plt.ylabel('Consumo (kWh)')
plt.show()

lw = 2
plt.scatter(X, y, color="darkorange", label="data")
plt.scatter(X_test, y_pred_test, color="navy", lw=lw, label="RBF Model")
plt.xlabel("data")
plt.ylabel("Consumption (kWh)")
plt.title("Energy Consumption")
plt.legend()
plt.show()




#GRID SEARCH

parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}

svr = SVR()
clf = GridSearchCV(svr, parameters)
clf.fit(X,y)
clf.best_params_

