#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:30:26 2022

@author: kunyu
"""
import numpy as np
import pandas as pd
from numpy.linalg import eigh
from scipy.stats import norm
from scipy.stats import t
from scipy import stats
from scipy import optimize
import pandas_datareader.data  as pdr
import datetime as dt

df = pd.read_csv('DailyPrices.csv')

def return_calculate(data,method = 'DISCRETE',dateColume = 'Date'):
    vars = data.drop(dateColume,axis = 1)
    p = np.matrix(vars)
    n = p.shape[0]
    m = p.shape[1]
    p2 = np.zeros((n-1,m))
    
    for i in range(n-1):
        for j in range(m):
            p2[i][j] = p[i+1][j] / p[i][j]
    
    if method.upper() == 'DISCRETE':
        p2 -= 1
    elif method.upper() == 'LOG':
        p2 = np.log(p2)
    else:
        print('method should be LOG or DISCRETE')
    return p2
        
#Calculate the arithmetic returns for INTC
INTC_prices = df['INTC']
INTC_return = return_calculate(pd.DataFrame(df[['Date','INTC']]),'DISCRETE','Date')
aapl_prices = df['INTC']
aapl_return = return_calculate(pd.DataFrame(df[['Date','INTC']]),'DISCRETE','Date')
#Remove the mean from the series so that the mean(INTC)=0
INTC_mean = np.mean(INTC_return)
INTC_adj = INTC_return - INTC_mean
INTC_std = np.std(INTC_adj)

#Calculate VaR
#1.Using a normal distribution
VaR1_ret = norm.ppf(0.05, 0,INTC_std)
VaR1 = INTC_prices[0] * VaR1_ret
print("VaR is %.2f using normal distribution" % VaR1)

#2. Using a normal distribution with an Exponentially Weighted variance (λ = 0. 94)

ew_var = pd.DataFrame(INTC_adj).ewm(alpha = 0.06).var()
ew_std = np.sqrt(ew_var).iloc[-1,:]
VaR2_ret = norm.ppf(0.05,0,ew_std)
VaR2 = INTC_prices[0] * VaR2_ret
print("VaR is %.2f using normal distribution with Exponentially Weighted variance" % VaR2)

#3. Using a MLE fitted T distribution
def t_ll(params):
    std , n = params[0], params[1]
    
    negLL = -np.sum( stats.t.logpdf(INTC_return, df=n, scale=std) )
    return negLL

model_t = optimize.minimize(t_ll, np.array([1,1]))
print("results of t_mle:")
print(model_t)
mle_std = model_t.x[0]
mle_n = model_t.x[1]
VaR3_ret = t.ppf(0.05,mle_n, loc = 0,scale = mle_std)
VaR3 = INTC_prices[0] * VaR3_ret
print("VaR is %.2f using normal distribution" % VaR3)

#4.Historic Simulation

a = int(0.05 * len(INTC_prices))
np.array(INTC_prices).sort()
PV = np.sum(INTC_prices)
VaR4 = PV - INTC_prices[a]

print("VaR is %.2f using historic simulation" % VaR4)

#download new data
def getData(stocks,start,end):
    stockData = pdr.get_data_yahoo(stocks,start = start, end = end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    returns = returns.dropna()
    meanReturns = returns.mean()
    
    return returns,meanReturns

endDate = dt.datetime.now()
startDate = '2022-01-15'
returns, meanReturns = getData('INTC',start = startDate,end = endDate)

print(returns)
print('The mean of returns is %.2f' % meanReturns)


