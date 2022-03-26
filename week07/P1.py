#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 18:38:40 2022

@author: kunyu
"""

import pandas as pd
import numpy as np
from datetime import date
from math import exp, log, pi, sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt

CurrentDate = date(2022,3,13)
ExpirDate = date(2022,4,15)
TimeToMaturity = (ExpirDate - CurrentDate).days / 365


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
#closed form greeks
def delta(underlying, strike, ttm, rf, b, ivol,q, t = 'call'):
    d1 = (log(underlying/strike) + (b +ivol**2/2) * ttm/(ivol*sqrt(ttm)))
    d2 = d1 - ivol * sqrt(ttm)
    dfq = exp(-q * ttm)
    if t.lower() == "call":
        return dfq * norm.cdf(d1)
    else:
        return dfq * norm.cdf(d1) - 1

delta_call_closedform = delta(underlying,K, TimeToMaturity, rf,b, 0.2, q, t = 'call')
print('delta of call option using closed form: %.2f' %delta_call_closedform)
delta_put_closedform = delta(underlying,K, TimeToMaturity, rf,b, 0.2, q, t = 'put')
print('delta of put option using closed form: %.2f' %delta_put_closedform)

def gamma(underlying, strike, ttm, rf, b, ivol,q):
    d1 = (log(underlying/strike) + (b +ivol**2/2) * ttm/(ivol*sqrt(ttm)))
    return exp(-q*ttm) * norm.pdf(d1)/(underlying * ivol * (ttm ** 0.5)) 

gamma_closedform = gamma(underlying, K, TimeToMaturity, rf, b, 0.2,q)
print('gamma of call option using closed form : %.2f' % gamma_closedform)
print('gamma of put option using closed form : %.2f' % gamma_closedform)

# Vega 
def vega(underlying, strike, ttm, rf, b, ivol,q):
    d1 = (log(underlying/strike) + (b +ivol**2/2) * ttm/(ivol*sqrt(ttm)))
    return underlying * exp(-q * ttm) * norm.pdf (d1) *ttm ** 0.5

vega_closedform = vega(underlying, K, TimeToMaturity, rf, b, 0.2,q)
print('vega of call option using closed form : %.2f' % vega_closedform)
print('vega of put option using closed form : %.2f' % vega_closedform)

# Theta 
def theta (underlying, strike, ttm, rf, b, ivol,q, t = 'call'):
    d1 = (log(underlying/strike) + (b +ivol**2/2) * ttm/(ivol*sqrt(ttm)))
    d2 = d1 - ivol * sqrt(ttm)
    df = exp(-rf * ttm)
    dfq = exp(-q * ttm)
    if t.lower() == 'call':
        tmptheta =  ( -0.5 * underlying * dfq * norm.pdf(d1) * ivol /(ttm ** 0.5) + \
        1 * (q * underlying * dfq * norm.cdf (1* d1) \
        - rf * strike * df * norm.cdf(1 * d2 )))
    else:
        tmptheta = ( -0.5 * underlying * dfq * norm.pdf(d1) * ivol /(ttm ** 0.5) + \
        -1 * (q * underlying * dfq * norm.cdf (-1* d1) \
        - rf * strike * df * norm.cdf(-1 * d2 )))
        
    return tmptheta

theta_call_closedform = theta(underlying, K, TimeToMaturity, rf, b, 0.2,q, t = 'call')
theta_put_closedform = theta(underlying, K, TimeToMaturity, rf, b, 0.2,q, t = 'put')
print('theta of call option using closed form: %.2f' % theta_call_closedform)
print('theta of put option using closed form: %.2f' % theta_put_closedform)

def rho(underlying, strike, ttm, rf, b, ivol,q, t = 'call'):
    d1 = (log(underlying/strike) + (b +ivol**2/2) * ttm/(ivol*sqrt(ttm)))
    d2 = d1 - ivol * sqrt(ttm)
    df = exp(-rf*ttm)
    if t.lower() == 'call':
        return strike * ttm * df * 0.01 * norm.cdf(d2)
    else:
        return -strike * ttm * df * 0.01 * norm.cdf(-d2)
    
rho_call_closedform = rho(underlying, K, TimeToMaturity, rf, b, 0.2,q, t = 'call')
rho_put_closedform = rho(underlying, K, TimeToMaturity, rf, b, 0.2,q, t = 'put')
print('rho of call option using closed form: %.2f' % rho_call_closedform)
print('rho of put option using closed form: %.2f' % rho_put_closedform)
    
def carryRho(underlying, strike, ttm, rf, b, ivol,q, t = 'call'):
    d1 = (log(underlying/strike) + (b +ivol**2/2) * ttm/(ivol*sqrt(ttm)))
    d2 = d1 - ivol * sqrt(ttm)
    dfq = exp(-q * ttm)
    if t.lower() == 'call':
        return  ttm * underlying * dfq * norm.cdf(d1)
    else:
        return - ttm * underlying * dfq * norm.cdf(-d1)
    
carryRho_call_closedform = carryRho(underlying, K, TimeToMaturity, rf, b, 0.2,q, t = 'call')   
carryRho_put_closedform = carryRho(underlying, K, TimeToMaturity, rf, b, 0.2,q, t = 'put')
print('carryRho of call option using closed form: %.2f' % carryRho_call_closedform)     
print('carryRho of put option using closed form: %.2f' % carryRho_put_closedform)  
    
#finite difference derivative calculation
def delta_fdm_call(S, K, T, r, sigma, b, ds = 1e-5):
    delta = (gbsm(S+ds, K, T, r, b, sigma, t = 'call') - gbsm(S, K, T, r, b, sigma, t = 'call'))/ds
    return delta

def delta_fdm_put(S, K, T, r, sigma, b, ds = 1e-5):
    delta = (gbsm(S+ds, K, T, r, b, sigma, t = 'put') - gbsm(S, K, T, r, b, sigma, t = 'put'))/ds
    return delta

delta_call_fd = delta_fdm_call(underlying, K, TimeToMaturity, rf, 0.2, b, ds = 1e-5)
print('delta of call option using forward difference method: %.2f' % delta_call_fd)

delta_put_fd = delta_fdm_put(underlying, K, TimeToMaturity, rf, 0.2, b, ds = 1e-5)
print('delta of put option using forward difference method: %.2f' % delta_put_fd)

def gamma_fdm(S, K, T, r, sigma,b, ds = 1e-5):
   gamma = (gbsm(S+2*ds, K, T, r,b, sigma, t = 'call') - 2*gbsm(S+ds, K, T, r, b,sigma, t = 'call')+
                   gbsm(S, K, T, r,b, sigma, t = 'call') )/ (ds**2)
   return gamma

gamma_fd =gamma_fdm(underlying, K, TimeToMaturity, rf, b, 0.2, ds = 1e-5)
print('gamma of call option using forward difference method: %.2f' % gamma_fd)
print('gamma of put option using forward difference method: %.2f' % gamma_fd)

def vega_fdm(S, K, T, r, b, sigma, dv=1e-5):
    
    vega = (gbsm(S, K, T, r, b, sigma+dv) - gbsm(S, K, T, r, b, sigma))/dv
    return vega
    
vega_fd = vega_fdm(underlying, K, TimeToMaturity, rf, b, 0.2, dv = 1e-5)   
print('vega of call option using forward difference method: %.2f' % vega_fd)
print('vega of put option using forward difference method: %.2f' % vega_fd)

def theta_call_fdm(S, K, T, r, b, sigma, dt = 1e-5):
    theta = -(gbsm(S, K, T+dt, r, b, sigma, t = 'call') - gbsm(S, K, T, r, b, sigma, t = 'call'))/dt
    return theta
    
def theta_put_fdm(S, K, T, r, b, sigma, dt=1e-5):
    theta = -(gbsm(S, K, T+dt, r, b, sigma, t = 'put') - gbsm(S, K, T, r, b, sigma, t = 'put'))/dt
    return theta

theta_call_fd = theta_call_fdm(underlying, K, TimeToMaturity, rf, b, 0.2, dt = 1e-5)
theta_put_fd = theta_put_fdm(underlying, K, TimeToMaturity, rf, b, 0.2, dt = 1e-5)
print('theta of call option using forward difference method: %.2f' % theta_call_fd)
print('theta of put option using forward difference method: %.2f' % theta_put_fd)

def rho_call_fdm(S, K, T, r, b, sigma, dr=1e-5):
    rho = (gbsm(S, K, T, r+dr, b, sigma,t = 'call') - gbsm(S, K, T, r, b, sigma, t = 'call'))/dr
    return rho
  
def rho_put_fdm(S, K, T, r, b, sigma, dr=1e-5):
    rho = (gbsm(S, K, T, r+dr, b ,sigma, t = 'put') - gbsm(S, K, T, r, b, sigma,t = 'put'))/dr
    return rho

rho_call_fd = rho_call_fdm(underlying, K, TimeToMaturity, rf, b, 0.2, dr = 1e-5)
rho_put_fd = rho_put_fdm(underlying, K, TimeToMaturity, rf, b, 0.2, dr = 1e-5)
print('rho of call option using forward difference method: %.2f' % rho_call_fd)
print('rho of put option using forward difference method: %.2f' % rho_put_fd)

def carryrho_call_fdm(S, K, T, r, b, sigma, db=1e-5):
    carryrho = (gbsm(S, K, T, r, b+db, sigma,t = 'call') - gbsm(S, K, T, r, b, sigma, t = 'call'))/db
    return carryrho
  
def carryrho_put_fdm(S, K, T, r, b, sigma, db=1e-5):
    carryrho = (gbsm(S, K, T, r, b+db ,sigma, t = 'put') - gbsm(S, K, T, r, b, sigma,t = 'put'))/db
    return carryrho

carryrho_call_fd = carryrho_call_fdm(underlying, K, TimeToMaturity, rf, b, 0.2, db = 1e-5)
carryrho_put_fd = carryrho_put_fdm(underlying, K, TimeToMaturity, rf, b, 0.2, db = 1e-5)
print('carry rho of call option using forward difference method: %.2f' % carryrho_call_fd)
print('carry rho of put option using forward difference method: %.2f' % carryrho_put_fd)