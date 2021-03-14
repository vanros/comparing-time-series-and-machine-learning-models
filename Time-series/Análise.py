# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:05:37 2021

@author: vanessa_rodrigues
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from matplotlib.dates import DateFormatter, MonthLocator, DayLocator, WeekdayLocator
import matplotlib.dates as mdates
import statsmodels.api as sm

import numpy as np
import matplotlib as mpl




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


data_monthly = data

data_monthly = data_monthly.resample('M').sum()

plt.plot(data.index, data.data);

# salvar a decomposicao em result</em>
result = seasonal_decompose(data, extrapolate_trend = 'freq')
result_monthly = seasonal_decompose(data_monthly)
result_monthly.plot()

plt.plot(data_monthly.index)

result = seasonal_decompose(train, model='additive')
result.plot()
plt.show()

print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)

seasonal = result.seasonal
trend = result.trend

result.plot()
###### WEEKLY #####################################
plt.plot(data_weekly);
dtFmt = mdates.DateFormatter('%m-%Y')
plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting'
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.xticks(rotation=45)
plt.show()

result = seasonal_decompose(data_weekly, model='multiplicative')
result.plot()
plt.show()


plt.plot(result.seasonal)
dtFmt = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_locator(WeekdayLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.ylabel('Seasonal')
plt.xticks(rotation=90)
plt.show()


################## MENSAL ###########################################

plt.plot(data_monthly);
dtFmt = mdates.DateFormatter('%m-%Y')
#plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting'
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.xticks(rotation=35)
plt.ylabel('Consumo (kWh)')
plt.show()

result = seasonal_decompose(data_monthly, model='additive')
result.plot()
plt.show()


plt.plot(result.seasonal)
dtFmt = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.ylabel('Seasonal')
plt.xticks(rotation=45)
plt.show()




#################################################################
plt.plot(result.seasonal)
dtFmt = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_locator(WeekdayLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.ylabel('Seasonal')
plt.xticks(rotation=90)
plt.show()

plt.plot(data_weekly)
dtFmt = mdates.DateFormatter('%m-%Y')
plt.gca().xaxis.set_major_locator(MonthLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.ylabel('Consumo (kWh)')
plt.xticks(rotation=35)
plt.show()

plt.plot(result.trend)
dtFmt = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_locator(WeekdayLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.xticks(rotation=90)
plt.show()

plt.plot(result.resid)
dtFmt = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_locator(WeekdayLocator()) # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) #
plt.xticks(rotation=90)
plt.show()


stl = STL(data_weekly)
res = stl.fit()

fig = res.plot()

fig = plt.subplots(2, 1)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



plot_acf(data, lags=50)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plot_pacf(data, lags=50)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

import matplotlib

matplotlib.rcParams.update({'font.size': 10})
figsize = (4,3) # same ratio, bigger text
fig,ax = plt.subplots(2, 1, figsize=figsize)
sm.graphics.tsa.plot_acf(data, lags=50, ax=ax[0])
sm.graphics.tsa.plot_pacf(data, lags=50, ax=ax[1])
plt.show()


#os resíduos mostram períodos de alta variabilidade nos períodos de abril de 2018 e 2019
# A sazonalidade parece ser razoável






# Prepare data
data['year'] = [d.year for d in data.index]
data['month'] = [d.strftime('%b') for d in data.index]
years = data['year'].unique()

# Prep Colors
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
plt.figure(figsize=(16,12), dpi= 80)
for i, y in enumerate(years):
    if i > 0:        
        plt.plot(data['month'], data=data.loc[data.year==y, :], color=mycolors[i], label=y)
        plt.text(data.loc[data.year==y, :].shape[0]-.9, data.loc[data.year==y, 'value'][-1:].values[0], y, fontsize=12, color=mycolors[i])

# Decoration
plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='Consumption', xlabel='Month')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot of Energy Consumption Time Series", fontsize=20)
plt.show()









#Unit root Test 
# KPSS test
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

kpss_test(data)



ts_t_adj = data - data.shift(1)
ts_t_adj = ts_t_adj.dropna()
ts_t_adj.plot()

data_diff = ts_t_adj
data_diff = ts_t_adj - ts_t_adj.shift(7)
data_diff = data_diff.dropna()
ts_t_adj.plot()

matplotlib.rcParams.update({'font.size': 10})
figsize = (4,3) # same ratio, bigger text
fig,ax = plt.subplots(2, 1, figsize=figsize)
sm.graphics.tsa.plot_acf(data, lags=50, ax=ax[0])
sm.graphics.tsa.plot_pacf(data, lags=50, ax=ax[1])
plt.show()





from pandas import read_csv
from matplotlib import pyplot
from numpy import polyfit
# fit polynomial: x^2*b1 + x*b2 + ... + bn
X = [i%7 for i in range(0, len(data))]
y = data.values
degree = 6
coef = polyfit(X, y, degree)
print('Coefficients: %s' % coef)
# create curve
curve = list()
for i in range(len(X)):
	value = coef[-1]
	for d in range(degree):
		value += X[i]**(degree-d) * coef[d]
	curve.append(value)
# plot curve over original data
pyplot.plot(data.values)
pyplot.plot(curve, color='red', linewidth=3)
pyplot.show()

data.info()

data.rolling(120).mean().plot(figsize=(20,10), linewidth=5, fontsize=10)

plt.figure(figsize=(15,5))
plt.plot(data)
plt.ylabel('Consumo')

from statsmodels.tsa.stattools import adfuller
def test_stationarity(data):
    p_val=adfuller(data)[1]
    if p_val >= 0.05:
        print("Time series data is not stationary. Adfuller test pvalue={}".format(p_val))
    else:
        print("Time series data is stationary. Adfuller test pvalue={}".format(p_val))
test_stationarity(data_weekly) 