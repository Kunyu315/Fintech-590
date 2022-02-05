import pandas as pd
from scipy import stats
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

df = pd.read_csv('problem2.csv')

y = df.y
x = df.x
#MLE: Normal distribution
def norm_ll(params):
    intercept, beta, std = params[0], params[1], params[2]
    #y_pred = intercept + beta*x
    e = y - intercept - beta*x
    negLL = -np.sum( stats.norm.logpdf(e, loc=0, scale=std) )
    return negLL

model_normal = optimize.minimize(norm_ll, np.array([1,1,1]), method='L-BFGS-B')
print("results of normal_mle:")
print(model_normal)
#calculate AIC

AIC_Nor = 2 * 3 + 2 * model_normal.fun
print("AIC of normal distribution is %d" % AIC_Nor)

#MLE: T distribution
def t_ll(params):
    intercept, beta, std , n = params[0], params[1], params[2], params[3]
    e = y - intercept - beta*x
    negLL = -np.sum( stats.t.logpdf(e, df=n, loc=0, scale=std) )
    return negLL

model_t = optimize.minimize(t_ll, np.array([1,1,1,1]), method='L-BFGS-B')
print("results of t_mle:")
print(model_t)
#calculate AIC
AIC_T = 2 * 4 + 2 * model_t.fun
print("AIC of T distribution is %d" % AIC_T)