# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:56:57 2020

@author: Propietario
"""

import numpy as np

def smape(forecast, real):
    n = len(forecast)
    array = [abs(forecast[i]-real[i])/((abs(real[i])+abs(forecast[i]))/2) for i in range(n)]
    value = 1/n*np.sum(array)
    return value

def mae(forecast, real):
    n = len(forecast)
    array = [abs(forecast[i]-real[i]) for i in range(n)]
    value = 1/n*np.sum(array)
    return value

def r2(forecast, real):
    res = forecast-real
    res_ss = np.sum(res**2)
    tot_ss = np.sum(real**2)
    value = 1-res_ss/tot_ss
    return value

def rmse(forecast, real):
    res = forecast-real
    res_2 = res**2
    value = np.sqrt(np.mean(res_2))
    return value
    