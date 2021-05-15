# -*- coding: utf-8 -*-
"""
Created on Fri May 14 19:29:32 2021

@author: 91998
"""

import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

# ticker symbols of DJIA constituents
symbols = [ 'MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT',
            'CVX', 'CSCO', 'KO', 'GS', 'HD', 'HON', 'IBM', 'INTC',
            'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PG', 'CRM',
            'TRV', 'UNH', 'VZ', 'V', 'WBA', 'WMT', 'DIS']


period = {"startdate": '2021-1-1', "enddate": '2021-3-31'}
maxplotcount = 10         
iterations = 5
pertubation_variance = 0.1

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
        
# Calculate percentage returns
lag_levels = levels.shift(1)        
returns = levels.div(lag_levels, axis = 0)    
returns.dropna(inplace = True)
returns = returns - 1
returns = returns * 100
    
# Compute correlation matrix
corrmatrix = returns.corr()    


eigen = np.linalg.eigvals(corrmatrix)
plotcount = min( len(eigen), maxplotcount )
eigen = list(map(np.linalg.norm, eigen ))
eigen.sort()
eigen = eigen[::-1]
eigen = eigen[0:plotcount]


fig, ax = plt.subplots()
ax.plot(range(plotcount), eigen, linestyle='dotted' )

numsymbols = len(symbols)

for i in range(iterations):
    Pertubation = np.random.normal(0, pertubation_variance, (numsymbols,numsymbols))
    np.fill_diagonal(Pertubation, 0)
    Pertubation = pd.DataFrame( Pertubation, columns = symbols, index = symbols )
    corrmatrix_new = corrmatrix + Pertubation
    eigen_new = np.linalg.eigvals(corrmatrix_new)
    #print(eigen_new, "\n")
    eigen_new = list(map(np.linalg.norm, eigen_new ))
    eigen_new.sort()
    eigen_new = eigen_new[::-1]
    print(eigen_new, "\n")
    eigen_new = eigen_new[0:plotcount]

    
    ax.plot(range(plotcount), eigen_new, linestyle='dotted')
                                  

ax.set_ylabel('Eigenvalue')
ax.set_xlabel('Index')

plt.grid(False)
plt.xticks(range(plotcount))
plt.plot()
plt.title('Eigenvalues under matrix pertubation')