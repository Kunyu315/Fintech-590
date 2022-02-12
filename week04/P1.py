#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:24:22 2022

@author: kunyu
"""
import numpy as np
import random

#Classical Brownian Motion


def CBM(t,P0,nsim):
    Pt_sim = []
    for i in range(nsim):
        r = np.random.randn(t)
        Pt = P0 + sum(r)
        Pt_sim.append(Pt)
    print("The Expected Value of Classical Brownian Motion is %.2f" % np.mean(Pt_sim))
    print("The Standard Deviation of Classical Brownian Motion is %.2f" % np.std(Pt_sim))
    
    
CBM(1,5,12000)

#Arithmetic Return
def ArithReturn(t,P0,nsim):
    Pt_sim = []
    for i in range(nsim):
        r = np.random.randn(t)
        Pt = P0
        for j in range(t):
            Pt = Pt*(1+r[j])
        Pt_sim.append(Pt)
    print("The Expected Value of Arithmetic Return is %.2f" % np.mean(Pt_sim))
    print("The Standard Deviation of Arithmetic Return is %.2f" % np.std(Pt_sim))

ArithReturn(1,5,12000)

# Geometric Brownian Motion
def GBM(t,P0,nsim):
    Pt_sim = []
    for i in range(nsim):
        r = np.random.randn(t)
        Pt = P0
        for j in range(t):
            Pt = Pt * np.exp(r[j])
        Pt_sim.append(Pt)
    print("The Expected Value of Geometric Brownian Motion is %.2f" % np.mean(Pt_sim))
    print("The Standard Deviation of Geometric Brownian Motion is %.2f" % np.std(Pt_sim))

GBM(1,5,12000)