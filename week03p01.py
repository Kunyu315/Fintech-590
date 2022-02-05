#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 21:38:46 2022

@author: kunyu
"""
import pandas as pd
import numpy as np
from numpy.linalg import eigh

import matplotlib.pyplot as plt

df = pd.read_csv('DailyReturn.csv')
df = df.drop(df.columns[[0]],axis = 1)

#calculate cov(x,y)
def ewma_cov_pairwise_pd(x, y, alpha):
    x = x.mask(y.isnull(), np.nan)
    y = y.mask(x.isnull(), np.nan)
    covariation = (x - x.mean()) * (y - y.mean()).dropna()
    ret = covariation.ewm(alpha = alpha).mean().iloc[-1]
    return ret
    
#calculate covariance matrix
def ewma_cov_pd(rets, alpha):   
    stocks = rets.columns[1:]
    n = len(stocks)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov[i, j] = cov[j, i] = ewma_cov_pairwise_pd(
                rets.iloc[:, i], rets.iloc[:, j], alpha=alpha)
    return pd.DataFrame(cov, columns=stocks, index=stocks)


#plot pca cumulative variance explained
def plot_pca(matrix):
    egnvalues, egnvectors = eigh(matrix)
    total_egnvalues = sum(egnvalues)
    var_exp = [(i/total_egnvalues) for i in sorted(egnvalues, reverse=True)]
    cum_sum_exp = np.cumsum(var_exp)
    plt.bar(range(0,len(var_exp)), var_exp, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_exp)), cum_sum_exp, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    

#lambda1 = 0.5
cov_pd_1 = ewma_cov_pd(df,0.5)
print(cov_pd_1)
plot_pca(cov_pd_1)

#lambda2 = 0.7
cov_pd_2 = ewma_cov_pd(df,0.3)
print(cov_pd_2)
plot_pca(cov_pd_2)

#lambda3 = 0.9
cov_pd_3 = ewma_cov_pd(df,0.1)
print(cov_pd_3)
plot_pca(cov_pd_3)

#lambda4 = 0.97
cov_pd_4 = ewma_cov_pd(df,0.03)
print(cov_pd_4)
plot_pca(cov_pd_4)

#lambda5 = 0.99
cov_pd_5 = ewma_cov_pd(df,0.01)
print(cov_pd_5)
plot_pca(cov_pd_5)
