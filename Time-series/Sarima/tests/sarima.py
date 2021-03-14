# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:40:46 2020

@author: vanessa_rodrigues
"""
import pandas as pd
import numpy as np
import numpy.matlib as npm
from numpy import empty
from dateutil.parser import parse
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.api as smtsa
from pmdarima.arima.utils import nsdiffs
from pmdarima.arima import auto_arima

dataset = pd.read_csv('energy_consumption_days.csv')

dataset = dataset.drop('Unnamed: 0', axis=1)

plt.figure(figsize=(10, 4))
plt.plot(dataset['Date'], dataset['Consumption (kWh)'])
plt.xlabel('Data')
plt.ylabel('Consumo (kWh)')

#Autocorrelação
plot_acf(dataset['Consumption (kWh)']);

#Autocorrelação parcial
plot_pacf(dataset['Consumption (kWh)']);

dataset_train = dataset[0:591]
dataset_test = dataset[591:738]

modelo_arma = smtsa.ARMA(dataset_train['Consumption (kWh)'], order=(3,0)).fit()

modelo_previsto = modelo_arma.predict(start=591, end=737)

plt.figure(figsize=(10, 4))
#plt.plot(dataset_train['Date'], dataset_train['Consumption (kWh)'])
plt.plot(dataset_test['Date'], dataset_test['Consumption (kWh)'])
plt.plot(dataset_test['Date'], modelo_previsto, 'r.')


modelo_arima = smtsa.ARIMA(dataset_train['Consumption (kWh)'].values, order=(1,1,2)).fit()

modelo_arima.plot_predict(26,35);
plt.plot(np.linspace(0,9,10), dataset_test['Consumption (kWh)'])

#Aplicando sazonalidade
 #(P,D,Q, m) m =número de observações por ciclo
D = nsdiffs(dataset_train['Consumption (kWh)'].values, m=2, max_D=12, test='ch')

modelo_sarima = auto_arima(dataset_test['Consumption (kWh)'].values, start_p=0, start_q=0, max_p=6, max_q=6, d=1, D=1, 
                                                                     start_Q =1, start_P=1, max_P=4, max_Q=4, m=2,
                                                                     seasonal=True, trace=True, error_action='ignore', suppress_warnings=True,
                                                                     stepwise=False) 

modelo_sarima.fit(dataset_train['Consumption (kWh)'].values)
valores_preditos = modelo_sarima.predict(n_periods=10)
plt.plot(valores_preditos)
plt.plot(np.linspace(0,10,147), dataset_test['Consumption (kWh)'])



#There are three trend elements that require configuration.

#p: Trend autoregression order.
#d: Trend difference order.
#q: Trend moving average order.

#There are four seasonal elements that are not part of ARIMA that must be configured; they are:

#P: Seasonal autoregressive order.
#D: Seasonal difference order.
#Q: Seasonal moving average order.
#m: The number of time steps for a single seasonal period.
#	SARIMA(p,d,q)(P,D,Q)m


#Grid Search

















































