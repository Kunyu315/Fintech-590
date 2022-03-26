#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:39:08 2022

@author: kunyu
"""

import pandas as pd
import numpy as np
from datetime import date
from math import exp, log, pi, sqrt
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import datetime

portfolio = pd.read_csv('problem2.csv')
currentDate = datetime.datetime(2022,2,25)
divDate = datetime.datetime(2022,3,15)
rf = 0.0025
S = 164.85
divAmt = [1.0]
divDays = (divDate - currentDate).days
divTimes = [divDays * 3]


returns = pd.read_csv('DailyReturn.csv')
returns = returns['AAPL']
aapl_std = returns.std()

#portfolio['ExpirationDate'] = pd.to_datetime(portfolio['ExpirationDate'])

vol = []

def bt_american(t, underlying, strike, ttm, rf, b, ivol, N):
    dt = ttm / N
    u = exp(ivol*sqrt(abs(dt)))
    d = 1 / u
    pu = (exp(b*dt) - d)/(u-d)
    pd = 1.0 - pu
    df = exp(-rf*dt)
    if t.lower() == 'call':
        z = 1
    else:
        z = -1
    nNodes = int((N+1)*(N+2)/2)
    optionValues = [0] * nNodes
    for j in range(N,-1,-1):
        for i in range(j,-1,-1):
            idx = int((j)*(j+1)/2) + i 
            price = underlying * (u**i) * (d**(j-i))
            optionValues[idx] = max(0,z*(price-strike))
            
            if j < N:
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[int((j+1)*(j+2)/2)+i+1] + pd*optionValues[int((j+1)*(j+2)/2)+i]) )
    return optionValues[0]

def bt_american_d(t, underlying, strike, ttm, rf, div, divTimes, ivol, N):
    if len(div) == 0 or len(divTimes) == 0:
        return bt_american('call', underlying, strike, ttm, rf,rf,ivol,N)
    elif divTimes[0] > N:
        return bt_american('call', underlying, strike, ttm, rf,rf,ivol,N)
    dt = ttm / N
    u = exp(ivol*sqrt(dt))
    d = 1 / u
    pu = (exp(rf*dt) - d)/(u-d)
    pd = 1.0 - pu
    df = exp(-rf*dt)
    if t.lower() == 'call':
        z = 1
    else:
        z = -1
    nDiv = len(divTimes)
    nNodes = int((divTimes[0]+1) * (divTimes[0]+2) / 2)
    optionValues = [0] * nNodes
    for j in range(divTimes[0], -1, -1):
        for i in range(j,-1,-1):
            idx = int(j * (j+1) /2) + i 
            price = underlying*(u**i)*(d**(j-i))  
            
            if j < divTimes[0]:
                optionValues[idx] = max(0,z*(price-strike))
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[int((j+1)*(j+2)/2)+i+1] + pd*optionValues[int((j+1)*(j+2)/2)+i]) )
            else:
                valNoExercise = bt_american_d('call', price-div[0], strike, ttm-divTimes[0]*dt, rf, div[1:nDiv], [m-divTimes[0] for m in divTimes[1:nDiv]], ivol, N-divTimes[0])
                valExercise =  max(0,z*(price-strike))
                optionValues[idx] = max(valNoExercise,valExercise)
    return optionValues[0]


#calculate implied vol
def cal_impliedVol_bt(data, i):
    s = data.iloc[i,:]
    
    Expiration = s['ExpirationDate']
    
    Type = s['Type']
    Strike = s['Strike']
    Price = s['CurrentPrice']
    OptionType = s['OptionType']
    
    if Type == 'Option':
        Expiration = datetime.datetime.strptime(Expiration,'%m/%d/%Y')
        ttm = (Expiration - currentDate).days / 365
        N = int((Expiration - currentDate).days * 3)
    
        volatility_candidates = np.arange(0.1, 1.0, 0.01)
        price_differences = np.zeros_like(volatility_candidates)
    
        observed_price = Price
    
        for i in range(len(volatility_candidates)):
    
            candidate = volatility_candidates[i]
    
            price_differences[i] = observed_price - bt_american_d(OptionType, S,Strike, ttm, rf, divAmt, divTimes,candidate ,N)
        
        idx = np.argmin(abs(price_differences))
        implied_volatility = volatility_candidates[idx]
        return implied_volatility

l = portfolio.shape[0]

for i in range(l):
    if portfolio.iloc[i,1] == ['Stock']:
        vol.append(None)
    else:
        
        iv = cal_impliedVol_bt(portfolio,i)
        vol.append(iv)

portfolio['ivol'] = vol

#calculate delta
def delta_bt(t, S, K, T, r, Div,DivTimes, ivol, N, ds = 1e-5):
    if t.lower() == 'call':
        delta = (bt_american_d('call', S+ds,K,T,r,Div,DivTimes,ivol,N)-bt_american_d('call', S,K,T,r,Div,DivTimes,ivol,N))/ds
    else:
        delta = (bt_american_d('put', S+ds,K,T,r,Div,DivTimes,ivol,N)-bt_american_d('put', S,K,T,r,Div,DivTimes,ivol,N))/ds
    return delta

Delta = []
def cal_delta(data, i):
    s = data.iloc[i,:]
    
    Expiration = s['ExpirationDate']
    
    Type = s['Type']
    Strike = s['Strike']
    Price = s['CurrentPrice']
    OptionType = s['OptionType']
    Holding = s['Holding']
    
    if Type == 'Option':
        Expiration = datetime.datetime.strptime(Expiration,'%m/%d/%Y')
        ttm = (Expiration - currentDate).days / 365
        N = int((Expiration - currentDate).days * 3) * Holding
        d = delta_bt(OptionType, S, Strike, ttm, rf, divAmt,divTimes, data['ivol'][i], N, ds = 1e-5)
    else:
        d = Holding
    return d

for i in range(l):
    Delta.append(cal_delta(portfolio, i))
portfolio['Delta'] = Delta

#simulate
nsim = 100
size = 10
simPrice = []

for i in range(nsim):
    simRet = norm.rvs(size = size, loc = 0, scale = aapl_std)
    simPrice.append(S * (simRet+1).prod())

values = []

def cal_simValues(data, i):
    
    s = data.iloc[i,:]
    
    Expiration = s['ExpirationDate']
    
    Type = s['Type']
    Strike = s['Strike']
    Price = s['CurrentPrice']
    OptionType = s['OptionType']
    
    if Type == 'Option':
        Expiration = datetime.datetime.strptime(Expiration,'%m/%d/%Y')
        ttm = (Expiration - currentDate).days / 365
        N = int((Expiration - currentDate).days * 3)
    
        for underlying in simPrice:
            v = bt_american_d(OptionType, underlying,Strike, ttm, rf, divAmt, divTimes, data['ivol'][i] ,N)
        return v
    else:
        return None
    
         
for i in range(l):
    values.append(cal_simValues(portfolio, i))
portfolio['Values'] = values

portfolio_type = portfolio['Portfolio'].unique()
def VaR_ES(x, alpha = 0.05):
    x.sort()
    n = alpha * len(x)
    iup = int(math.ceil(n))
    idn = int(math.floor(n))
    VaR = (x[iup] + x[idn]) / 2
    ES = np.mean(x[:idn])
    
    return -VaR, -ES


mean_sim = []
VaR_sim = []
ES_sim = []

for i in range(l):
    mean = portfolio.loc[i,'Values'].mean()
    VaR , ES= VaR_ES( simPrice[i:], 0.05)
    mean_sim.append(mean)
    VaR_sim.append(VaR)
    ES_sim.append(ES)
    
results = pd.DataFrame()

results['Mean'] = mean_sim
results['VaR'] = VaR_sim
results['ES'] = ES_sim
results.index = portfolio['Portfolio']
print(results)
    
    
    
    
    
    