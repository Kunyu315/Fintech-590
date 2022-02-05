#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:16:29 2022

@author: kunyu
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

#Problem2 OLS
df2 = pd.read_csv('problem2.csv')
x = (df2['x']).tolist()
y = (df2['y']).tolist()

X = sm.add_constant(x)
model = sm.OLS(y,X)
results = model.fit()
print(results.params)


error = [0]*100

beta0 = results.params[0]
beta1 = results.params[1]

for i in range(len(x)):
    error[i] = y[i]-beta0-beta1*x[i]

print(error)
#distribution
print(stats.normaltest(error))
print(stats.shapiro(error))

mean = np.mean(error)
print(mean)
print('error mean is %.3f' % mean)

median = np.median(error)
var = np.var(error)
print('error median is %.3f' % median)
print('error variance is %.3f' % var)
skewness = stats.skew(error)
print('error skewness is %.3f' % skewness)
kurtosis = stats.kurtosis(error)
print('error kurtosis is %.3f' % kurtosis) 

#MLE
