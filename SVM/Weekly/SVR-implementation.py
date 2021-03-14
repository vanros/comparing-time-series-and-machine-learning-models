# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:01:49 2021

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

data = pd.read_csv('series-todataframe.csv')

data = data.drop('Unnamed: 0', 1)

data = data.rename(columns={'var1(t-1)': 'X', 'var1(t)': 'y'})

X = data.X.values.reshape(-1,1)
y = data.y.values.reshape(-1,1)

mse_list_testError = []
rmse_list_testError = []

for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)
    svr = SVR(C = 10, epsilon = 0.1, gamma = 0.0001, kernel = 'rbf')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mse_list_testError.append(mse)
    rmse_list_testError.append(rmse)

print('MSE_TestError:',  mean(mse_list_testError))
print('RMSE_TestError:', mean(rmse_list_testError))

#GRID SEARCH

parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}

svr = SVR()
clf = GridSearchCV(svr, parameters)
clf.fit(X,y)
clf.best_params_