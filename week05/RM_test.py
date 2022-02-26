#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:17:20 2022

@author: kunyu
"""

import RM
import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt

#test Covariance estimation
df_return = pd.read_csv('DailyReturn.csv')
df_return = df_return.drop(df_return.columns[[0]],axis = 1)
assets = df_return.shape[1]

cov_return = RM.ewma_cov_pd(df_return,0.5)
print(cov_return)

#test Non PSD fixes for correlation matrices
n = 500
sigma = np.zeros(500*500)
sigma = sigma.reshape(500,500)
sigma.fill(0.9)

for i in range(n):
    sigma[i,i] = 1.0
sigma[0,1] = 0.7357
sigma[1,0] = 0.7357

W = np.identity(n)

hpsd = RM.higham_nearPD(sigma)
npsd = RM.near_psd(sigma)
norm_hpsd = RM.wgtNorm(hpsd-sigma,W)
norm_npsd = RM.wgtNorm(npsd-sigma,W)
print("n=500")
print("Distance near_psd() = %.2f " % norm_npsd)
print("Distance higham_nearPSD() = %.2f" % norm_hpsd)

#test Simulation Methods

def cov1(df,assets):
    cov1 = np.zeros(shape = (assets,assets))
    for i in range(assets):
        for j in range(assets):
            cov1[i][j] = np.corrcoef(df.transpose())[i][j] * np.sqrt((np.var(df))[i]) * np.sqrt((np.var(df))[j])
    return cov1
cov1_df = cov1(df_return,assets)
sim1 = RM.simulate_pca(assets, cov1_df,nsim =25000)

covar1 = np.cov(sim1.T)
norms1 = np.sum((covar1 - cov1_df)**2)
print("Norms 1 is %.5f" % norms1)

#test VaR calculation methods
df_prices = pd.read_csv('DailyPrices.csv')
INTC_prices = df_prices['INTC']
INTC_return = RM.return_calculate(pd.DataFrame(df_prices[['Date','INTC']]),'DISCRETE','Date')
aapl_prices = df_prices['INTC']
aapl_return = RM.return_calculate(pd.DataFrame(df_prices[['Date','INTC']]),'DISCRETE','Date')
#Remove the mean from the series so that the mean(INTC)=0
INTC_mean = np.mean(INTC_return)
INTC_adj = INTC_return - INTC_mean
INTC_std = np.std(INTC_adj)

VaR1 = RM.VaR_normal(INTC_prices, 0, INTC_std, 0.05)
print("VaR is %.2f using normal distribution" % VaR1)
VaR2 = RM.VaR_EW(INTC_prices, INTC_return, 0.05, 0)
print("VaR is %.2f using normal distribution with Exponentially Weighted variance" % VaR2)

#model_t = optimize.minimize(RM.t_ll(params = [1,1],ret = INTC_return), np.array([1,1]))
VaR3 = RM.VaR_mle(prices = INTC_prices, loc = 0, alpha = 0.05, ret = INTC_return)
print("VaR is %.2f using MLE T distribution" % VaR3)
VaR4 = RM.VaR_historical(INTC_prices, INTC_return, alpha = 0.05)
print("VaR is %.2f using historical data" % VaR4)

#test ES calculation
df_ES = pd.read_csv('problem1.csv')
mu, std = norm.fit(df_ES)
plt.hist(df_ES, bins = 25, density = True, alpha = 0.6, color = 'g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin,xmax, 100)

VaR_N, ES_N = RM.VaR_ES(norm.pdf(x,mu,std),0.05)
print('ES of fitted Normal Distribution is %.6f' % ES_N)
