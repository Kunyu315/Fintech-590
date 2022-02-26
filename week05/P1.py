#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:05:00 2022

@author: kunyu
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt
import math

df = pd.read_csv('/problem1.csv')

#Fit a Normal Distribution
mu, std = norm.fit(df)

plt.hist(df, bins = 25, density = True, alpha = 0.6, color = 'g')

xmin, xmax = plt.xlim()
x = np.linspace(xmin,xmax, 100)
p = norm.pdf(x,mu,std)
plt.plot(x,p,'k',linewidth = 2)
title = 'Fit a Normal Distribution'
plt.title(title)
plt.show()

# Fit a Generalized T distribution
plt.hist(df, bins = 25, density = True, alpha = 0.6, color = 'g')
d, loc, scale = t.fit(np.array(df,dtype=np.float128))
pdf_fitted = t.pdf(x, df = d, loc = loc, scale =scale)
plt.plot(x,pdf_fitted,'r',linewidth = 2)
title = 'Fit a Generalized T Distribution'
plt.title(title)
plt.show()


#Calculate the VaR and ES
def VaR_ES(x, alpha = 0.05):
    x.sort()
    n = alpha * len(x)
    iup = int(math.ceil(n))
    idn = int(math.floor(n))
    VaR = (x[iup] + x[idn]) / 2
    ES = x[:idn].mean()
    
    return -VaR, -ES

VaR_N, ES_N = VaR_ES(p,0.05)
VaR_T, ES_T = VaR_ES(pdf_fitted,0.05)
print('VaR of fitted Normal Distribution is %.6f' % VaR_N)
print('ES of fitted Normal Distribution is %.6f' % ES_N)
print('VaR of fitted T Distribution is %.2f' % VaR_T)
print('ES of fitted T Distribution is %.2f' % ES_T)

#overlay
plt.plot(x,p,'k',x,pdf_fitted,'r')
plt.axvline(VaR_N,color = 'k',label = 'VaR_N')
plt.axvline(VaR_T,color = 'r',label = 'VaR_T')
plt.axvline(ES_N,color = 'g',label = 'ES_N')
plt.axvline(ES_T,color = 'b',label = 'ES_T')
title = 'Overlay Two Distributions'
plt.title(title)
plt.legend()
plt.show()