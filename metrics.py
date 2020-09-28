# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:56:57 2020

@author: Propietario
"""

def smape(forecast, real):
    n = len(forecast)
    array = np.array(math.abs(forecast[i]-real[i])/((math.abs(real[i])+math.abs(forecast[i]))/2) for i in range(n)
    value = 1/n*np.sum(array)
    return value

def mae