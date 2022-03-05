#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:03:56 2022

@author: kunyu
"""

import pandas as pd
import numpy as np

import datetime
from math import exp, log, pi, sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.stats as st 
import math

data = pd.read_csv('problem3.csv')
DailyReturn = pd.read_csv('DailyReturn.csv')

currentPrice = 164.85
rf = 0.0025
q = 0.0053
b = rf - q
CurrentDate = datetime.datetime(2022,2,25)

def gbsm( underlying, strike, ttm, rf, b, ivol, t = 'call'):
    d1 = (log(underlying/strike) + (b +ivol**2/2) * ttm/(ivol*sqrt(ttm)))
    d2 = d1 - ivol * sqrt(ttm)
    
    if t.lower() == 'call':
        return underlying * exp((b-rf)*ttm) * norm.cdf(d1) - strike*exp(-rf*ttm)* norm.cdf(d2)
    else:
        return strike*exp(-rf*ttm)* norm.cdf(-d2) - underlying*exp((b-rf)*ttm)* norm.cdf(-d1)
    
data['implied vol'] = ''

def cal_impliedVol(i):
    s = data.iloc[i,:]
    
    Expiration = s['ExpirationDate']
    
    Type = s['Type']
    Strike = s['Strike']
    Price = s['CurrentPrice']
    OptionType = s['OptionType']
    
    if Type == 'Option':
        Expiration = datetime.datetime.strptime(Expiration,'%m/%d/%Y')
        ttm = (Expiration - CurrentDate).days / 365
    
        volatility_candidates = np.arange(0.1, 1.0, 0.0001)
        price_differences = np.zeros_like(volatility_candidates)
    
        observed_price = Price
    
        for i in range(len(volatility_candidates)):
    
            candidate = volatility_candidates[i]
    
            price_differences[i] = observed_price - gbsm(currentPrice, Strike, ttm, rf, b, candidate ,OptionType)
        
        idx = np.argmin(abs(price_differences))
        implied_volatility = volatility_candidates[idx]
        return implied_volatility

l = data.shape[0]

for i in range(l):
    iv = cal_impliedVol(i)
    data.iloc[i, -1] = iv


def cal_portfolioV(i):
    s = data.iloc[i,:]
    
    Expiration = s['ExpirationDate']
    
    Type = s['Type']
    Strike = s['Strike']
    Price = s['CurrentPrice']
    OptionType = s['OptionType']
    
    if Type == 'Option':
        Expiration = datetime.datetime.strptime(Expiration,'%m/%d/%Y')
        ttm = (Expiration - CurrentDate).days / 365
        ivol = s['implied vol']
        
        underlyingValues = np.arange(Strike-0.5, Strike+0.5, 0.001)
        PortfolioValues = [gbsm(j, Strike, ttm, rf, b, ivol, OptionType) for j in underlyingValues]
        return PortfolioValues


def plot_portfolio_value(i):
    s = data.iloc[i,:]
    
    Expiration = s['ExpirationDate']
    
    Type = s['Type']
    Strike = s['Strike']
    Price = s['CurrentPrice']
    OptionType = s['OptionType']
    
    if Type == 'Option':
        
        p = cal_portfolioV(i)
        underlyingValues = np.arange(Strike-0.5, Strike+0.5, 0.001)
        plt.plot(underlyingValues,p)
        plt.xlabel('underlying values')
        plt.ylabel('portfolio values')
        plt.title(s['Portfolio'] + ' ' +  OptionType)
        
plot_portfolio_value(0)
plot_portfolio_value(1)
#change index 1-15
        
#Fit a Normal distribution to AAPL returns
AAPL = DailyReturn['AAPL']
average = sum(AAPL) / len(AAPL)
AAPL = AAPL - average

mu, std = norm.fit(AAPL)

plt.hist(AAPL, bins = 25, density = True, alpha = 0.6, color = 'g')

xmin, xmax = plt.xlim()
x = np.linspace(xmin,xmax, 100)
p = norm.pdf(x,mu,std)
plt.plot(x,p,'k',linewidth = 2)
title = 'Fit a Normal Distribution'
plt.title(title)
plt.show()

#Simulate AAPL returns 10 days ahead
aapl_10 = st.norm.rvs(loc = 0, scale = std, size = 10)
AAPL = np.array(AAPL.tolist())

AAPL = AAPL.tolist()
for i in range(10):
    AAPL.append(aapl_10[i])

AAPL_mean = np.mean(AAPL)

def VaR_ES(x, alpha = 0.05):
    x.sort()
    n = alpha * len(x)
    iup = int(math.ceil(n))
    idn = int(math.floor(n))
    VaR = (x[iup] + x[idn]) / 2
    ES = np.mean(x[:idn])
    
    return -VaR, -ES

VaR_AAPL, ES_AAPL = VaR_ES(AAPL,0.05)
print('Mean is %.2f' % AAPL_mean )
print('VaR is %.2f' % VaR_AAPL)
print('ES is %.2f' % ES_AAPL)