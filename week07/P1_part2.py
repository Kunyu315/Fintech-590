#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:32:07 2022

@author: kunyu
"""

#binomial tree valuation
import pandas as pd
import numpy as np
from datetime import date
from math import exp, log, pi, sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt

CurrentDate = date(2022,3,13)
ExpirDate = date(2022,4,15)
TimeToMaturity = (ExpirDate - CurrentDate).days / 365
Div = [0.88]
DivDate = date(2022,4,11)
DivDays = (DivDate - CurrentDate).days
ttm = 365 * TimeToMaturity
DivTimes = [DivDays * 3]
N = int(ttm * 3)

def gbsm( underlying, strike, ttm, rf, b, ivol, t = 'call'):
    d1 = (log(underlying/strike) + (b +ivol**2/2) * ttm/(ivol*sqrt(ttm)))
    d2 = d1 - ivol * sqrt(ttm)
    
    if t.lower() == 'call':
        return underlying * exp((b-rf)*ttm) * norm.cdf(d1) - strike*exp(-rf*ttm)* norm.cdf(d2)
    else:
        return strike*exp(-rf*ttm)* norm.cdf(-d2) - underlying*exp((b-rf)*ttm)* norm.cdf(-d1)
    

rf = 0.0025
q = 0.0053
b = rf - q
underlying = 165
K =165

def bt_american(t, underlying, strike, ttm, rf, b, ivol, N):
    dt = ttm / N
    u = exp(ivol*sqrt(dt))
    d = 1 / u
    pu = (exp(b*dt) - d)/(u-d)
    pd = 1.0 - pu
    df = exp(-rf*dt)
    if t.lower() == 'call':
        z = 1
    else:
        z = -1
    nNodes = int((N+1)*(N+2)/2)
    optionValues = [0] * nNodes
    for j in range(N,-1,-1):
        for i in range(j,-1,-1):
            idx = int((j)*(j+1)/2) + i 
            price = underlying * (u**i) * (d**(j-i))
            optionValues[idx] = max(0,z*(price-strike))
            
            if j < N:
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[int((j+1)*(j+2)/2)+i+1] + pd*optionValues[int((j+1)*(j+2)/2)+i]) )
    return optionValues[0]

def bt_american_d(t, underlying, strike, ttm, rf, div, divTimes, ivol, N):
    if len(div) == 0 or len(divTimes) == 0:
        return bt_american('call', underlying, strike, ttm, rf,rf,ivol,N)
    elif divTimes[0] > N:
        return bt_american('call', underlying, strike, ttm, rf,rf,ivol,N)
    dt = ttm / N
    u = exp(ivol*sqrt(dt))
    d = 1 / u
    pu = (exp(rf*dt) - d)/(u-d)
    pd = 1.0 - pu
    df = exp(-rf*dt)
    if t.lower() == 'call':
        z = 1
    else:
        z = -1
    nDiv = len(divTimes)
    nNodes = int((divTimes[0]+1) * (divTimes[0]+2) / 2)
    optionValues = [0] * nNodes
    for j in range(divTimes[0], -1, -1):
        for i in range(j,-1,-1):
            idx = int(j * (j+1) /2) + i 
            price = underlying*(u**i)*(d**(j-i))  
            
            if j < divTimes[0]:
                optionValues[idx] = max(0,z*(price-strike))
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[int((j+1)*(j+2)/2)+i+1] + pd*optionValues[int((j+1)*(j+2)/2)+i]) )
            else:
                valNoExercise = bt_american_d('call', price-div[0], strike, ttm-divTimes[0]*dt, rf, div[1:nDiv], [m-divTimes[0] for m in divTimes[1:nDiv]], ivol, N-divTimes[0])
                valExercise =  max(0,z*(price-strike))
                optionValues[idx] = max(valNoExercise,valExercise)
    return optionValues[0]

bt_call = bt_american_d('call', underlying,K,TimeToMaturity,rf,Div,DivTimes,0.2,N)
bt_put = bt_american_d('put', underlying,K,TimeToMaturity,rf,Div,DivTimes,0.2,N)

print('call option value using binomial tree valuation: %.2f' % bt_call)
print('put option value using binomial tree valuation: %.2f' % bt_put)

#calculate greeks
def delta_bt_call(S, K, T, r, Div,DivTimes, N, ds = 1e-5):
    delta = (bt_american_d('call', S+ds,K,T,r,Div,DivTimes,0.2,N)-bt_call)/ds
    return delta

def delta_bt_put(S, K, T, r, Div,DivTimes, N, ds = 1e-5):
    delta = (bt_american_d('put', S+ds,K,T,r,Div,DivTimes,0.2,N)-bt_put)/ds
    return delta

delta_bt_c = delta_bt_call(underlying,K,TimeToMaturity,rf,Div,DivTimes,N)
delta_bt_p = delta_bt_put(underlying,K,TimeToMaturity,rf,Div,DivTimes,N)
print('delta of call option using binomial tree valuation: %.2f' % delta_bt_c)
print('delta of put option using binomial tree valuation: %.2f' % delta_bt_p)

def gamma_bt(S, K, T, r, Div, DivTimes, N, ds = 1e-5):
    gamma = (bt_american_d('call', S+ds,K,T,r,Div,DivTimes,0.2,N) + bt_american_d('call', S-ds,K,T,r,Div,DivTimes,0.2,N)-2*bt_call)/(ds**2)
    return gamma
gamma_bt = gamma_bt(underlying,K,TimeToMaturity,rf,Div,DivTimes,N)
print('gamma of call option using binomial tree valuation: %.2f' % gamma_bt)
print('gamma of put option using binomial tree valuation: %.2f' % gamma_bt)
    
def vega_bt(S, K, T, r, Div,DivTimes, N, dv=1e-5):
    
    vega = (bt_american_d('call', S,K,T,r,Div,DivTimes,0.2+dv,N) - bt_american_d('call', S,K,T,r,Div,DivTimes,0.2,N))/dv
    return vega
    
vega_bt = vega_bt(underlying,K,TimeToMaturity,rf,Div,DivTimes,N, dv= 1e-5)   
print('vega of call option using binomial tree valuation: %.2f' % vega_bt)
print('vega of put option using binomial tree valuation: %.2f' % vega_bt) 

def theta_call_bt(S, K, T, r, Div,DivTimes, N, dt = 1e-5):
    theta = -(bt_american_d('call', S,K,T+dt,r,Div,DivTimes,0.2,N) - bt_american_d('call', S,K,T,r,Div,DivTimes,0.2,N))/dt
    return theta
    
def theta_put_bt(S, K, T, r, Div,DivTimes, N, dt=1e-5):
    theta = -(bt_american_d('put', S,K,T+dt,r,Div,DivTimes,0.2,N) - bt_american_d('put', S,K,T,r,Div,DivTimes,0.2,N))/dt
    return theta

theta_call_bt = theta_call_bt(underlying,K,TimeToMaturity,rf,Div,DivTimes,N, dt = 1e-5)
theta_put_bt = theta_put_bt(underlying,K,TimeToMaturity,rf,Div,DivTimes,N, dt = 1e-5)
print('theta of call option using binomial tree valuation: %.2f' % theta_call_bt)
print('theta of put option using binomial tree valuation: %.2f' % theta_put_bt)

def rho_call_bt(S, K, T, r,Div,DivTimes,N, dr=1e-5):
    rho = (bt_american_d('call', S,K,T,r+dr,Div,DivTimes,0.2,N) - bt_american_d('call', S,K,T,r,Div,DivTimes,0.2,N))/dr
    return rho
  
def rho_put_bt(S, K, T, r, Div,DivTimes, N, dr=1e-5):
    rho = (bt_american_d('put', S,K,T,r+dr,Div,DivTimes,0.2,N) - bt_american_d('put', S,K,T,r,Div,DivTimes,0.2,N))/dr
    return rho

rho_call_bt = rho_call_bt(underlying,K,TimeToMaturity,rf,Div,DivTimes,N, dr = 1e-5)
rho_put_bt = rho_put_bt(underlying,K,TimeToMaturity,rf,Div,DivTimes,N, dr = 1e-5)
print('rho of call option using binomial tree valuation: %.2f' % rho_call_bt)
print('rho of put option using binomial tree valuation: %.2f' % rho_put_bt)



    