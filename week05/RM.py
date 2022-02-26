#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:32:41 2022

@author: kunyu
"""
import numpy as np
import pandas as pd
import math
from numpy.linalg import eigh
from timeit import timeit
import functools
import time
from scipy.stats import norm, t

from scipy import stats
from scipy import optimize
import pandas_datareader.data  as pdr
import datetime as dt
import matplotlib.pyplot as plt

#1. Covariance estimation techniques.
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


#2. Non PSD fixes for correlation matrices
def near_psd(a,epsilon = 1e-3):
    n = a.shape[0]
    invSD = None
    out = a.copy()
    
    invSD = np.matrix(np.diag(1.0/np.sqrt(np.diag(out))))
    out = invSD * out * invSD
    
    vals,vecs = np.linalg.eig(out)
    vals = np.maximum(vals,epsilon)
    vals = vals.reshape(n,1)
    t = 1.0/(vecs * vecs * vals)
    t = t.tolist()
    T = np.identity(n)
    for i in range(n):
        T[i,i] = np.sqrt(t[i][0])
    l = np.identity(n)
    vals = vals.tolist()
    for i in range(n):
        l[i,i] = np.sqrt(vals[i][0])
    
    B = T * vecs * l
    out = B * B.T
    
    if invSD is not None:
        invSD = np.diag(1.0/np.diag(invSD))
        out = invSD * out * invSD
        
    return out

def chol_psd(a):
    a = np.array(a,float)
    L = np.zeros_like(a)
    n,_ = np.shape(a)
    for j in range(n):
        for i in range(j,n):
            if i == j:
                L[i,j] = np.sqrt(a[i,j] - np.sum(L[i,:j]**2))
            else:
                    L[i,j] = (a[i,j]-np.sum(L[i,:j]*L[j,:j]))/L[j,j]
    return L

def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def wgtNorm(A,W):
    W05 = np.sqrt(W)
    W05 = W05 * A * W05
    return np.sum(W05 * W05)


def higham_nearPD(pc, W = None, epsilon = 1e-9, maxIter=100,tol = 1e-9):
    n = pc.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = pc.copy()
    i = 1
    while(i <= maxIter):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
        norm = wgtNorm(Yk-pc,W)
        minEigval = np.min(np.linalg.eig(Yk)[0])
        
        if minEigval > -epsilon:
            break
        i += 1
        
    if i < maxIter:
        print("Converged in %d iterations." % i)
    else:
        print("Convergence failed after %d iterations." %(i-1))
        
    return Yk


#3. Simulation Methods


def simulate_pca(assets, a,nsim, nval = None):
    egnvalues, egnvectors = eigh(a)
    total_egnvalues = sum(egnvalues)
    egnvalues = sorted(egnvalues,reverse = True)
    vectors = [0] * assets
    for i in range(assets):
        vectors[i] = egnvectors[assets - i -1]
    
    posv = []
    for i in range(len(egnvalues)):
        if egnvalues[i] >=1e-8:
            posv.append(egnvalues[i])
            
    if nval is not None:
        if nval < len(posv):
            posv = posv[:nval]
            
    egnvalues = posv
    vectors = vectors[:][:len(posv)]
    vectors = np.array(vectors).reshape(assets,len(posv))
            
    print("Simulating with %d PC Factors: %d%%  total variance explained" % (len(posv), (sum(egnvalues)/total_egnvalues*100 )))
    B = np.dot(vectors , np.diag(np.sqrt(egnvalues)))
    
    m = len(egnvalues)
    r = np.random.randn(m,nsim)
    return (np.dot(B,r)).T


#4. VaR calculation methods (all discussed)

def return_calculate(data,method = 'DISCRETE',dateColume = 'Date'):
    vars = data.drop(dateColume,axis = 1)
   
    p2 = vars.copy()
    if method.upper() == 'DISCRETE':
        for j in vars.columns[:]:
            p2[j] = vars[j] / vars[j].shift() - 1
        
    elif method.upper() == 'LOG':
        for j in vars.columns[:]:
            p2[j] = np.log(vars[j] / vars[j].shift())
    else:
        print('method should be LOG or DISCRETE')
    return p2.iloc[1:,:]

#Calculate VaR
#(1)Using a normal distribution
def VaR_normal(prices, loc, scale, alpha):
    VaR = prices[0] * norm.ppf(alpha, loc = loc, scale = scale)
    return VaR

#(2)Using a normal distribution with an Exponentially Weighted variance (λ = 0. 94)

def VaR_EW(prices, ret, alpha, loc):
    ew_var = pd.DataFrame(ret).ewm(alpha = alpha).var()
    ew_std = np.sqrt(ew_var).iloc[-1,:]
    VaR = prices[0] * norm.ppf(alpha, loc = loc, scale = ew_std)
    return VaR
      

#(3)Using a MLE fitted T distribution

def VaR_mle(prices, loc, alpha, ret):
    def t_ll(params):
        n,std = params
        return -np.sum(stats.t.logpdf(ret,df = n, scale = std))
    params = np.array([1,1])
    model_t = optimize.minimize(t_ll, params)
    
    mle_std = model_t.x[1]
    mle_n = model_t.x[0]
    VaR = prices[0] * t.ppf(alpha, mle_n, loc = loc, scale = mle_std)
    return VaR
                                
#(4)Historic Simulation
def VaR_historical(prices, ret, alpha):
    return prices[0] * np.quantile(ret, q = alpha)

#5. ES calculation

def VaR_ES(x, alpha = 0.05):
    x.sort()
    n = alpha * len(x)
    iup = int(math.ceil(n))
    idn = int(math.floor(n))
    VaR = (x[iup] + x[idn]) / 2
    ES = x[:idn].mean()
    
    return -VaR, -ES