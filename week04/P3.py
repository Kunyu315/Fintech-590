#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:56:56 2022

@author: kunyu
"""
import numpy as np
import pandas as pd

df = pd.read_csv('DailyPrices.csv')
Portfolio = pd.read_csv('Portfolio.csv')

def return_calculate(data,method = 'DISCRETE'):
    
    p = np.matrix(data)
    n = p.shape[0]
    m = p.shape[1]
    p2 = np.zeros((n-1,m))
    
    for i in range(n-1):
        for j in range(m):
            p2[i][j] = p[i+1][j] / p[i][j]
    
    if method.upper() == 'DISCRETE':
        p2 -= 1
    elif method.upper() == 'LOG':
        p2 = np.log(p2)
    else:
        print('method should be LOG or DISCRETE')
    return p2
        
def P_VaR(port):
    data = Portfolio[Portfolio.Portfolio == port]
    num = len(data)
    PV = 0
    r = []
    prices = []

    for i in range(num):
        PV += np.array(data.Holding)[i] * np.sum(df[np.array(data.Stock)[i]])
        prices.append(df[np.array(data.Stock)[i]])
        ret = return_calculate(pd.DataFrame(df[np.array(data.Stock)[i]]),'DISCRETE')
        r.append(ret)
    np.array(r).sort()
    np.array(prices).sort()
    a = int(len(prices) * 0.05)
    VaR = PV - prices[a]
    return VaR
P_VaR('A')
P_VaR('B')
P_VaR('C')

def VaR_total(P):
    num = len(P)
    PV = 0
    r = []
    prices = []
    for i in range(num):
        PV += np.array(P.Holding)[i] * np.sum(df[np.array(P.Stock)[i]])
        prices.append(df[np.array(P.Stock)[i]])
        ret = return_calculate(pd.DataFrame(df[np.array(P.Stock)[i]]))
        r.append(ret)
    np.array(r).sort()
    np.array(prices).sort()
    a = int(len(prices) * 0.05)
    VaR = PV - prices[a]
    return VaR

VaR_total(Portfolio)