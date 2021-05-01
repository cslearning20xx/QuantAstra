# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:58:58 2021

@author: Chetan Sharma
"""


import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

# ticker symbols of DJIA constituents
symbols = [ 'MMM', 'AXP','AMGN', 'AAPL', 'BA', 'CAT',
            'CVX', 'CSCO', 'KO', 'GS', 'HD', 'HON', 'IBM', 'INTC',
            'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PG', 'CRM',
            'TRV', 'UNH', 'VZ', 'V', 'WBA', 'WMT', 'DIS']

# x-axis for plotting euler characteristics curves
corrthresholds = np.linspace( 0,1,80)

# Time periods considered for analysis, 1 month period at every 6 month interval
periods = [ 
            {"startdate": '2019-2-20', "enddate": '2019-3-20'},
            {"startdate": '2019-8-20', "enddate": '2019-9-20'},
            {"startdate": '2020-2-20', "enddate": '2020-3-20'},
            {"startdate": '2020-8-20', "enddate": '2020-9-20'},
            {"startdate": '2021-2-20', "enddate": '2021-3-20'}
          ]


# Given the correlation matrix and threshold, computes the Euler characteristic as
# EC = V - E, where V is the number of vertices and E is the number of edges in the
# correlation network which have edge weight less than threshold
def getEC(corrmatrix, threshold):
    arr = corrmatrix.to_numpy()
    arr = np.absolute(arr)
    edges = arr[ arr <= threshold ]
    EC =  corrmatrix.shape[0] - len(edges)
    return EC

fig, ax = plt.subplots()

for idx, period in enumerate(periods):
    all = []
    for tickersymbol in symbols:           
        tickerData = yf.Ticker(tickersymbol)
        # get the time series and pick the Close value
        tickerDf = tickerData.history( period = '1d', start = period["startdate"], end= period["enddate"])        
        tickerDf = tickerDf[['Close']]
        tickerDf.rename(columns = {'Close':tickersymbol }, inplace = True)
        if tickerDf.shape[0] > 10:
            all.append(tickerDf)
        
    levels = pd.concat( all, axis = 1)
    levels.dropna(inplace = True)
    periods[idx].update({ "levels": levels})
        
    # Calculate percentage returns
    lag_levels = levels.shift(1)        
    returns = levels.div(lag_levels, axis = 0)    
    returns.dropna(inplace = True)
    returns = returns - 1
    returns = returns * 100
    
    periods[idx].update({ "returns": returns})
    
    # Compute correlation matrix
    corrmatrix = returns.corr()    
    periods[idx].update({ "corrmatrix": corrmatrix})

    EC = []
    for corrthreshold in corrthresholds:
        EC.append(getEC(corrmatrix, corrthreshold))
        
    periods[idx]["EC"] = EC
    label = str(period["startdate"]) + ":" + str(period["enddate"])
    ax.plot(corrthresholds, EC, linestyle='dotted', label = label )

ax.set_ylabel('Euler Characteristic')
ax.set_xlabel('Correlation')

plt.plot()
plt.title('Euler Characteristics curves over time periods for DJIA')
plt.legend()