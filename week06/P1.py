#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 22:55:37 2022

@author: kunyu
"""

import pandas as pd
import numpy as np
from datetime import date
from math import exp, log, pi, sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt

CurrentDate = date(2022,2,25)
ExpirDate = date(2022,3,18)
TimeToMaturity = (ExpirDate - CurrentDate).days / 365
print(TimeToMaturity)

def gbsm( underlying, strike, ttm, rf, b, ivol, t = 'call'):
    d1 = (log(underlying/strike) + (b +ivol**2/2) * ttm/(ivol*sqrt(ttm)))
    d2 = d1 - ivol * sqrt(ttm)
    
    if t == 'call':
        return underlying * exp((b-rf)*ttm) * norm.cdf(d1) - strike*exp(-rf*ttm)* norm.cdf(d2)
    else:
        return strike*exp(-rf*ttm)* norm.cdf(-d2) - underlying*exp((b-rf)*ttm)* norm.cdf(-d1)
    

rf = 0.0025
q = 0.0053
b = rf - q
underlying = 165


impliedVol = np.arange(0.1, 0.8, 0.01)

calls = [gbsm(underlying, 165, TimeToMaturity, rf ,b ,ivol) for ivol in impliedVol]
puts = [gbsm(underlying, 165, TimeToMaturity, rf ,b ,ivol, t = 'put') for ivol in impliedVol]
plt.plot(impliedVol, calls, label='Call Value')
plt.plot(impliedVol, puts, label='Put Value')
plt.xlabel('$\sigma$')
plt.ylabel(' Value')
plt.legend()