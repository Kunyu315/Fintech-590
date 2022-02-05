#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 20:00:00 2022

@author: kunyu
"""

import numpy as np
import pandas as pd
from numpy.linalg import eigh
import time

df = pd.read_csv('DailyReturn.csv')
df = df.drop(df.columns[[0]],axis = 1)
assets = df.shape[1]

def Pearson_corr(data):
    corr = np.corrcoef(data.transpose())
    return corr

Pearson_corr_df = Pearson_corr(df)


def Pearson_var(data):
    var = np.var(data)
    return var

Pearson_var_df =Pearson_var(df)

def EW_corr(data):
    corr = data.ewm(alpha = 0.03).corr()
    return corr

EW_corr_df = EW_corr(df).iloc[-assets:,:]

def EW_var(data):
    var = data.ewm(alpha = 0.03).var()
    return var

EW_var_df = EW_var(df).iloc[-1,:]
#covariance matrix 1 :Pearson correlation + var()
def cov1():
    cov1 = np.zeros(shape = (assets,assets))
    for i in range(assets):
        for j in range(assets):
            cov1[i][j] = Pearson_corr_df[i][j] * np.sqrt(Pearson_var_df[i]) * np.sqrt(Pearson_var_df[j])
    return cov1
cov1_df = cov1()

#covariance matrix 2 :Pearson correlation + EW var()
def cov2():
    cov2 = np.zeros(shape = (assets,assets))
    for i in range(assets):
        for j in range(assets):
            cov2[i][j] = Pearson_corr_df[i][j] * np.sqrt(EW_var_df[i]) * np.sqrt(EW_var_df[j])
    return cov2
cov2_df = cov2()
    
    
#covariance matrix 3 : EW correlation + var()
def cov3():
    cov3 = np.zeros(shape = (assets,assets))
    for i in range(assets):
        for j in range(assets):
            cov3[i][j] = EW_corr_df.iloc[i,j] * np.sqrt(Pearson_var_df[i]) * np.sqrt(Pearson_var_df[j])
    return cov3
cov3_df = cov3()
    

#covariance matrix 4 :EW correlation + EW var()
def cov4():
    cov4 = np.zeros(shape = (assets,assets))
    for i in range(assets):
        for j in range(assets):
            cov4[i][j] = EW_corr_df.iloc[i,j] * np.sqrt(EW_var_df[i]) * np.sqrt(EW_var_df[j])
    return cov4
cov4_df = cov4()

def simulate_pca(a,nsim, nval = None):
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


W = np.identity(assets)

#cov1_df
#sim1
starttime = time.time()    
sim1 = simulate_pca(cov1_df,25000)
endtime = time.time()
sim1_times = endtime-starttime
print("sim1 took: %.10f seconds" %sim1_times)
covar1 = np.cov(sim1.T)
norms1 = np.sum((covar1 - cov1_df)**2)
print("Norms 1 is %.5f" % norms1)

#sim2
starttime = time.time()
sim2 = simulate_pca(cov1_df,25000,nval = 11)
endtime = time.time()
sim2_times = endtime-starttime
print("sim2 took: %.10f seconds" %sim2_times)
covar2 = np.cov(sim2.T)
norms2 = np.sum((covar2 - cov1_df)**2)
print("Norms 2 is %.5f" % norms2)

#sim3
starttime = time.time()
sim3 = simulate_pca(cov1_df,25000,nval = 12)
endtime = time.time()
sim3_times = endtime-starttime
print("sim3 took: %.10f seconds" %sim3_times)
covar3 = np.cov(sim3.T)
norms3 = np.sum((covar3 - cov1_df)**2)
print("Norms 3 is %.5f" % norms3)

#sim4
starttime = time.time()
sim4 = simulate_pca(cov1_df,25000,nval = 4)
endtime = time.time()
sim4_times = endtime - starttime
print("sim4 took: %.10f seconds" % sim4_times)
covar4 = np.cov(sim4.T)
norms4 = np.sum((covar4 - cov1_df)**2)
print("Norms 4 is %.5f" % norms4)

#sim5
starttime = time.time()
sim5 = simulate_pca(cov1_df,25000,nval = 3)     
endtime = time.time()
sim5_times = endtime - starttime
print("sim5 took: %.10f seconds" % sim5_times)
covar5 = np.cov(sim5.T)
norms5 = np.sum((covar5 - cov1_df)**2)
print("Norms 5 is %.5f" % norms5)
    
#cov2_df
#sim1
starttime = time.time()
sim1 = simulate_pca(cov2_df,25000)
endtime = time.time()
sim1_times = endtime - starttime
print("sim1 took: %.10f seconds" % sim1_times)
covar1 = np.cov(sim1.T)
norms1 = np.sum((covar1 - cov2_df)**2)
print("Norms 1 is %.5f \n" % norms1)

#sim2

starttime = time.time()
sim2 = simulate_pca(cov2_df,25000,nval = 12)
endtime = time.time()
sim2_times = endtime - starttime
print("sim2 took: %.10f seconds" % sim2_times)
covar2 = np.cov(sim2.T)
norms2 = np.sum((covar2 - cov2_df)**2)
print("Norms 2 is %.5f \n" % norms2)

#sim3
starttime = time.time()
sim3 = simulate_pca(cov2_df,25000,nval =3)
endtime = time.time()
sim3_times = endtime - starttime
print("sim3 took: %.10f seconds" % sim3_times)
covar3 = np.cov(sim3.T)
norms3 = np.sum((covar3 - cov2_df)**2)
print("Norms 3 is %.5f \n" % norms3)

#cov3_df

#sim1
starttime = time.time()
sim1 = simulate_pca(cov3_df,25000)
endtime = time.time()
sim1_times = endtime - starttime
print("sim1 took: %.10f seconds" % sim1_times)
covar1 = np.cov(sim1.T)
norms1 = np.sum((covar1 - cov3_df)**2)
print("Norms 1 is %.5f \n" % norms1)

#sim2
starttime = time.time()
sim2 = simulate_pca(cov3_df,25000,nval = 10)
endtime = time.time()
sim2_times = endtime - starttime
print("sim2 took: %.10f seconds" % sim2_times)
covar2 = np.cov(sim2.T)
norms2 = np.sum((covar2 - cov3_df)**2)
print("Norms 2 is %.5f \n" % norms2)

#sim3
starttime = time.time()
sim3 = simulate_pca(cov3_df,25000,nval = 3)
endtime = time.time()
sim3_times = endtime - starttime
print("sim3 took: %.10f seconds" % sim3_times)
covar3 = np.cov(sim3.T)
norms3 = np.sum((covar3 - cov3_df)**2)
print("Norms 3 is %.5f \n" % norms3)
    
#cov4_df

#sim1
starttime = time.time()
sim1 = simulate_pca(cov4_df,25000)
endtime = time.time()
sim1_times = endtime - starttime
print("sim1 took: %.10f seconds" % sim1_times)
covar1 = np.cov(sim1.T)
norms1 = np.sum((covar1 - cov4_df)**2)
print("Norms 1 is %.5f \n" % norms1)

#sim2
starttime = time.time()
sim2 = simulate_pca(cov4_df,25000,nval = 10)
endtime = time.time()
sim2_times = endtime - starttime
print("sim2 took: %.10f seconds" % sim2_times)
covar2 = np.cov(sim2.T)
norms2 = np.sum((covar2 - cov4_df)**2)
print("Norms 2 is %.5f \n" % norms2)

#sim3
starttime = time.time()
sim3 = simulate_pca(cov4_df,25000,nval = 3)
endtime = time.time()
sim3_times = endtime - starttime
print("sim3 took: %.10f seconds" % sim3_times)
covar3 = np.cov(sim3.T)
norms3 = np.sum((covar3 - cov4_df)**2)
print("Norms 3 is %.5f \n" % norms3)    

