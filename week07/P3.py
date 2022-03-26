#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 23:51:28 2022

@author: kunyu
"""

import pandas as pd
import numpy as np
from datetime import date
import datetime
from math import exp, log, pi, sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

ffr = pd.read_csv('F-F_Research_Data_Factors_daily.csv')
ffm = pd.read_csv('F-F_Momentum_Factor_daily.csv')

returns = pd.read_csv('DailyReturn.csv')

rf = 0.0025

stocks = ['AAPL', 'FB', 'UNH', 'MA', 'MSFT', 'NVDA', 'HD', 'PFE', 'AMZN', 'BRK_B',
          'PG', 'XOM', 'TSLA', 'JPM', 'V', 'DIS', 'GOOGL', 'JNJ', 'BAC', 'CSCO']
ff = pd.merge(ffr,ffm, on = 'Date')

ff['Date'] = pd.to_datetime(ff['Date'], format = '%Y%m%d')
ff.columns = ['Date','MktRF', 'SMB', 'HML', 'RF', 'Mom']
ff[['MktRF', 'SMB', 'HML', 'RF', 'Mom']] = ff[['MktRF', 'SMB', 'HML', 'RF', 'Mom']] / 100

returns['Date'] = pd.to_datetime(returns['Date'])
returns = pd.merge(returns, ff, on = 'Date')

params = []
returns = returns.rename(columns = {'BRK-B':'BRK_B'})

for i in stocks:
    returns[i] = returns[i]- returns['RF']
    result = sm.ols( formula = i +' ~ MktRF + SMB + HML + Mom', data = returns).fit()
    params.append(result.params[1:])
    
E_rf = ff['RF'].mean()
E_factors = ff[["MktRF", "SMB", "HML", "Mom"]].mean(axis=0)
l = len(params)

E_returns = []
for i in range(l):
    e_r = log(abs((params[i] * E_factors).sum())) + 1
    e_r = (e_r + E_rf) * 255
    E_returns.append(e_r)
#annual_cov = (log(returns[stocks] + 1)).cov() * 255
#print(annual_cov)

E_returns_result = pd.DataFrame()
E_returns_result['stocks'] = stocks
E_returns_result['expected annual return'] = E_returns
print(E_returns_result)
    