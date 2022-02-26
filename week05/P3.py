#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 23:29:40 2022

@author: kunyu
"""

import RM
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt

prices = pd.read_csv('DailyPrices.csv')
portfolio = pd.read_csv('portfolio.csv')

results = pd.DataFrame()

VaR_list = []
ES_list = []

def get_portfolio(data, prices):
    holdingsP = data['Holding'].tolist()
    stockP = data['Stock'].tolist()
    pricesP = prices[stockP]
    pricesP = pd.concat([prices['Date'],pricesP],axis = 1)
    retP = RM.return_calculate(pricesP)
    
    totalValueP= (pricesP.iloc[-1,1:] * holdingsP).sum()
    weightsP = pricesP.iloc[-1,1:] * holdingsP / totalValueP
    covarianceP = np.cov(retP.T)
    portfolioP_std = np.sqrt(np.dot(np.dot(weightsP.T, covarianceP),weightsP))
    
    return retP, portfolioP_std

def calculateVaR_ES( portfolioStd, dof, alpha = 0.05,portfolioReturn = 0,):
    VaR = np.sqrt((dof-2)/dof) * t.ppf(1-alpha,dof) * portfolioStd - portfolioReturn
    x_anu = t.ppf(alpha, dof)
    ES = -1/(alpha) * (1-dof)**-1 * (dof - 2 +x_anu**2) * t.pdf(x_anu, dof) * portfolioStd - portfolioReturn
    return VaR, ES


#A
dataA = portfolio[portfolio['Portfolio'] == 'A']
retA, portfolioA_std = get_portfolio(dataA, prices = prices)

d_A, loc, scale = t.fit(np.array(retA,dtype=np.float128))
VaR_A, ES_A = calculateVaR_ES(portfolioA_std, d_A, 0.05, 0)

VaR_list.append(VaR_A)
ES_list.append(ES_A)

#B

dataB = portfolio[portfolio['Portfolio'] == 'B']

retB, portfolioB_std = get_portfolio(dataB, prices = prices)
d_B, loc, scale = t.fit(np.array(retB,dtype=np.float128))
VaR_B, ES_B = calculateVaR_ES(portfolioB_std, d_B, 0.05, 0)

VaR_list.append(VaR_B)
ES_list.append(ES_B)
    
#C
dataC = portfolio[portfolio['Portfolio'] == 'C']

retC, portfolioC_std = get_portfolio(dataC, prices = prices)
d_C, loc, scale = t.fit(np.array(retC,dtype=np.float128))
VaR_C, ES_C = calculateVaR_ES(portfolioC_std, d_C, 0.05, 0)

VaR_list.append(VaR_C)
ES_list.append(ES_C)

#total
dataT = portfolio
retT, portfolioT_std = get_portfolio(dataT, prices = prices)
d_T, loc, scale = t.fit(np.array(retT,dtype=np.float128))
VaR_T, ES_T = calculateVaR_ES(portfolioT_std, d_T, 0.05, 0)

VaR_list.append(VaR_T)
ES_list.append(ES_T)

results['VaR'] = VaR_list
results['ES'] = ES_list
results.index = ['A','B','C','Total']
print(results)



