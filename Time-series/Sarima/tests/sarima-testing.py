# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:25:54 2021

@author: vanessa_rodrigues
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 22:29:14 2021

@author: vanessa_rodrigues
"""
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
import pmdarima as pm

series = pd.read_csv('C://Users//vanessa_rodrigues//Documents//Pesquisa-mestrado//Time-series//Sarima//city_day.csv')
series_delhi = series.loc[series['City'] == 'Delhi']
ts_delhi = series_delhi[['Date','AQI']]
#converting 'Date' column to type 'datetime' so that indexing can happen later
ts_delhi['Date'] = pd.to_datetime(ts_delhi['Date'])

ts_delhi.isnull().sum()
ts_delhi = ts_delhi.dropna()
ts_delhi.isnull().sum()

ts_delhi = ts_delhi.set_index('Date')

#monthly

ts_month_avg = ts_delhi['AQI'].resample('MS').mean()
ts_month_avg.plot(figsize = (15, 6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(ts_month_avg, model='additive')
fig = decomposition.plot()
plt.show()

#Thumb Rule for Statistical Tests –
#ADF: if the p-value is less than the critical value, the series is stationary
#OSCB: if the value is less than 0.64, the series is stationary

from statsmodels.tsa.stattools import adfuller

def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number  of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

print(adf_test(ts_month_avg))


#After running the ADF test on the time series, we obtain the following output.
# Since the p-value of 0.96 is greater than the critical value of 0.05,
# we can statistically confirm that the series is not stationary.
#Hence, we would do first-order differencing for the trend and 
#re-run the ADF test to check for stationarity.

ts_t_adj = ts_month_avg - ts_month_avg.shift(1)
ts_t_adj = ts_t_adj.dropna()
ts_t_adj.plot()


print(adf_test(ts_t_adj))

#The p-value is less than the critical value of 0.05. Hence we can confirm that the series is now trend stationary.

#Let us now move onto seasonal differencing. Since the data is showing an
# annual seasonality, we would perform the differencing at a lag 12, i.e yearly.

ts_s_adj = ts_t_adj - ts_t_adj.shift(12)
ts_s_adj = ts_s_adj.dropna()
ts_s_adj.plot()

#ACF stands for Auto Correlation Function and PACF stands for Partial Auto Correlation Function.

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(ts_s_adj)
matplotlib.pyplot.show()
plot_pacf(ts_s_adj)
matplotlib.pyplot.show()

#For ACF plot, initial spikes at lag = 1 and seasonal spikes at lag = 12,
# which means a probable AR order of 1 and seasonal AR order of 1
#For PACF plot, initial spikes at lag = 1 and seasonal spikes at lag = 12, 
#which means a probable MA order of 1 or 2 and seasonal MA order of 1
#SARIMA(p,d,q)x(P,D,Q)lag
#So, our probable SARIMA model equation can be: SARIMA(1,1,1)x(1,1,1)12 

#Grid Search

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
            mod = sm.tsa.statespace.SARIMAX(ts_s_adj['AQI'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue






