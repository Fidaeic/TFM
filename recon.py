# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:57:52 2020

@author: Fidae El Morer
"""
import pandas as pd
import numpy as np

def recon_median(df, nul, weeks=6):
    dataframe = df.copy()
    #Este método únicamente tendrá en cuenta la mediana de los días que se le haya introducido
    for n in nul:
        values = []
        for k in range(weeks):
            values.append(dataframe.loc[n-pd.Timedelta(value=(k+1)*7, unit='D')])
        dataframe.loc[n] = np.median(values)
    return dataframe

def recon_mean(df, nul, weeks=6):
    #Este método hace lo mismo que el anterior, solo que con la media
    dataframe = df.copy()
    for n in nul:
        values = []
        for k in range(weeks):
            values.append(dataframe.loc[n-pd.Timedelta(value=(k+1)*7, unit='D')])
        dataframe.loc[n] = np.mean(values)
    return dataframe

def recon_hybrid(df, nul, short='ARIMA', long='median', weeks=6):
    dataframe = df.copy()
    for n in nul:
        values = []
        for k in range(weeks):
            values.append(dataframe.loc[n-pd.Timedelta(value=(k+1)*7, unit='D')]) 

        if long == 'median':
            dataframe.loc[n] = np.median(values)
        elif long == 'mean':
            dataframe.loc[n] = np.mean(values)
            
        