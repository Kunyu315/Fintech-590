#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 23:04:13 2022

@author: kunyu
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

#Problem1
df1 = pd.read_csv('problem2.csv')
x = (df1['x']).tolist()
y = (df1['y']).tolist()

E_y = np.mean(y)
E_x = np.mean(x)
var_x = np.cov(x)

cov_xy = (np.cov(x,y))[0][1]

#conditional distribution
coefficient= cov_xy*(1/var_x)
intercept = E_y - coefficient * E_x

E_condition = "conditional distribution is %.2f + %.2f * x" %(intercept, coefficient)
print(E_condition)

#OLS
X = sm.add_constant(x)
model = sm.OLS(y,X)
results = model.fit()
#print(results.params)
#print(results.summary())
E_ols= "OLS equation is %.2f + %.2f * x" %(results.params[0],results.params[1])
print(E_ols)

