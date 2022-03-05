#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 22:55:00 2022

@author: kunyu
"""

import pandas as pd
import numpy as np

import datetime
from math import exp, log, pi, sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt


aapl = pd.read_csv('AAPL_Options.csv')
currentPrice = 164.85
rf = 0.0025
q = 0.0053
b = rf - q
CurrentDate = datetime.datetime(2022,2,25)
aapl['implied vol'] = ''
    
s = aapl.iloc[0,:]
stock = s['Stock']
Expiration = s['Expiration']
Expiration = datetime.datetime.strptime(Expiration,'%m/%d/%Y')
Type = s['Type']
Strike = s['Strike']
Price = s['Last Price']
ttm = (Expiration - CurrentDate).days / 365

N = norm.cdf

def gbsm( underlying, strike, ttm, rf, b, ivol, t = 'call'):
    d1 = (log(underlying/strike) + (b +ivol**2/2) * ttm/(ivol*sqrt(ttm)))
    d2 = d1 - ivol * sqrt(ttm)
    
    if t.lower()== 'call':
        return underlying * exp((b-rf)*ttm) * norm.cdf(d1) - strike*exp(-rf*ttm)* norm.cdf(d2)
    else:
        return strike*exp(-rf*ttm)* norm.cdf(-d2) - underlying*exp((b-rf)*ttm)* norm.cdf(-d1)



def cal_impliedVol(i):
    s = aapl.iloc[i,:]
    stock = s['Stock']
    Expiration = s['Expiration']
    Expiration = datetime.datetime.strptime(Expiration,'%m/%d/%Y')
    Type = s['Type']
    Strike = s['Strike']
    Price = s['Last Price']
    ttm = (Expiration - CurrentDate).days / 365
    
    volatility_candidates = np.arange(0.1,0.8, 0.0001)
    price_differences = np.zeros_like(volatility_candidates)
    
    observed_price = Price
    
    for i in range(len(volatility_candidates)):
    
        candidate = volatility_candidates[i]
    
        price_differences[i] = observed_price - gbsm(currentPrice, Strike, ttm, rf, b, candidate ,Type)
        
    idx = np.argmin(abs(price_differences))
    implied_volatility = volatility_candidates[idx]
    return implied_volatility
    #print('Implied volatility for option is:', implied_volatility)
    
l = aapl.shape[0]
for i in range(l):
    iv = cal_impliedVol(i)
    aapl['implied vol'][i] = iv
print(aapl)
    
#plot 
aapl_call = aapl[aapl['Type'] == 'Call']
aapl_call_strike = aapl_call['Strike']
appl_call_impliedvol = aapl_call['implied vol']
plt.plot(aapl_call_strike, aapl_call_strike)
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.title('Call Options')

aapl_put = aapl[aapl['Type'] == 'Put']
aapl_put_strike = aapl_put['Strike']
appl_put_impliedvol = aapl_put['implied vol']
plt.plot(aapl_put_strike, aapl_put_strike)
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.title('Put Options')