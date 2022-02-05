#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 21:13:05 2022

@author: kunyu
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# Plot 1: ACF: AR(1)  parameter = 0.9

ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1000)
#plt.plot(simulated_data_1);

plot_acf(simulated_data_1, alpha=1, lags=20);

# Plot 2 PACF AR(1)
plot_pacf(simulated_data_1, lags=20);


# Plot 3:ACF AR(2) paramter phi1 = 0.6, phi2 = 0.3
ma2 = np.array([1])
ar2 = np.array([1, -0.6, -0.3])
AR_object = ArmaProcess(ar2, ma2)
simulated_data_2 = AR_object.generate_sample(nsample=5000)
plot_acf(simulated_data_2, alpha=1, lags=20);

#Plot 4 PACF:AR(2)
plot_pacf(simulated_data_2, lags=20);

# Plot 5 ACF AR(3) paramter phi1 = 0.6, phi2 = 0.2, phi3 = 0.1
ma3 = np.array([1])
ar3 = np.array([1, -0.6, -0.3, -0.1])
AR_object = ArmaProcess(ar3, ma3)
simulated_data_3 = AR_object.generate_sample(nsample=5000)
plot_acf(simulated_data_3, alpha=1, lags=20);

# Plot 6 PACF: AR(3)
plot_pacf(simulated_data_3, lags=20);

# Plot 7: ACF MA(1) parameter: -0.9

ar4 = np.array([1])
ma4 = np.array([1, -0.9])
MA_object1 = ArmaProcess(ar4, ma4)
simulated_data_4 = MA_object1.generate_sample(nsample=1000)
plot_acf(simulated_data_4, lags=20);


#Plot 8:PACF MA(1) parameter: -0.9
plot_pacf(simulated_data_4, lags=20);

# Plot 9:ACF MA(2) paramter phi1 = 0.6, phi2 = 0.3
ma5 = np.array([1])
ar5 = np.array([1, -0.6, -0.3])
MA_object = ArmaProcess(ar5, ma5)
simulated_data_5 = MA_object.generate_sample(nsample=5000)
plot_acf(simulated_data_5, alpha=1, lags=20);

# Plot 10 PACF: MA(2)
plot_pacf(simulated_data_5, lags=20);

# Plot 11 ACF MR(3) paramter phi1 = 0.6, phi2 = 0.2, phi3 = 0.1
ma6 = np.array([1])
ar6 = np.array([1, -0.6, -0.3, -0.1])
AR_object = ArmaProcess(ar6, ma6)
simulated_data_6 = AR_object.generate_sample(nsample=5000)
plot_acf(simulated_data_6, alpha=1, lags=20);

# Plot 6 PACF: MA(3)
plot_pacf(simulated_data_6, lags=20);