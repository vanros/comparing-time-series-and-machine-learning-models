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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn.metrics import mean_absolute_error




data = pd.read_csv('C://Users//vanessa_rodrigues//Documents//Pesquisa-mestrado//SVM//Daily//series-todataframe.csv')

data = data.drop('Unnamed: 0', 1)

data = data.rename(columns={'var1(t-1)': 'X', 'var1(t)': 'y'})

X = data.X.values.reshape(-1,1)
y = data.y.values.reshape(-1,1)


scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.20, random_state=42)

rmse_list_testError = []
r2_list_testError = []
mae_list = []


rmse_list_trainError = []
r2_list_trainError = []
training_time_list =[]

predict_time_list = []

for i in range(30):
    regressor = RandomForestRegressor( n_estimators = 800, max_depth = 50, min_samples_leaf = 4, min_samples_split = 10, bootstrap= True, max_features = 'sqrt')

    t_start_training = time.time()
    regressor.fit(X_train, y_train)
    t_training = round( time.time()-t_start_training, 3) # the time would be round to 3 decimal in seconds
    training_time_list.append(t_training)
    
    t_start_predict =  time.time()
    y_pred_test = regressor.predict(X_test)
    t_predict = round( time.time()-t_start_predict, 3)
    predict_time_list.append(t_predict)
     
    y_pred_train = regressor.predict(X_train)
    
   
   
    
    y_pred_test_mape = y_pred_test.reshape(148,1)
    y_pred_train_mape = y_pred_train.reshape(590,1)

    rmse_list_testError.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2_list_testError.append(r2_score(y_test, y_pred_test))
    mae_list.append(mean_absolute_error(y_test, y_pred_test))
    
    rmse_list_trainError.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    r2_list_trainError.append(r2_score(y_train, y_pred_train))
    


print('RMSE_TestError:', mean(rmse_list_testError))
print('MAE:', mean(mae_list))

print('R-squared_TestError:', mean(r2_list_testError))

print('RMSE_TrainError:', mean(rmse_list_trainError))
print('R-squared_TrainError:', mean(r2_list_trainError))


print('training  Time (s)', mean(training_time_list))
print('predict  Time (s)', mean(predict_time_list))


y_pred_test = y_pred_test.reshape(148, 1)

plt.plot(scaler.inverse_transform(y_test), label='Teste ')
plt.plot(scaler.inverse_transform(y_pred_test), label='Predições')
plt.legend(framealpha=1, frameon=True);
plt.ylabel('Consumo (kWh)')
plt.show()

ax=data.plot()
test['Prediction'].plot(ax=ax,label='Predicted')


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


#Random Search

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)
# Use the random grid to search for best hyperparameters
rf = RandomForestRegressor(random_state = 42)

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error', 
                              cv = 3, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)
rf_random.fit(X_train, y_train);

rf_random.best_params_