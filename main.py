# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:30:18 2020

@author: Propietario
"""

from waTS import *
import pandas as pd
import itertools
import numpy as np

import os

df_flow = pd.read_csv("Data/flow.txt", parse_dates=[3], sep=';', header = None, skiprows=0).drop([0, 1, 2], axis=1)

recon_meths = ['mean', 'median', 'hybrid']

long_meths = ['mean', 'median']

short_meths = ['ARIMA', 'HW', 'KNN', 'RF', 'SVR']

corr = [True, False]
nn = [False, True]
models = [KNeighborsRegressor(), RandomForestRegressor(), SVR(), 'NN']

resample = True

if resample==True:
    stat=24*7
    hor=24
else:
    stat=96
    hor=96
weeks = 4

pipe = Pipeline(df_flow)
pipe.wrangle(plot=0)

for rec in recon_meths:
    print("Recon meth is: ", str(rec))
    if rec=='hybrid':
        for short in short_meths:
            print("Short recon meth is: ", str(short))
            for long in long_meths:
                print("Long recon meth is: ", str(long))
                for cor in corr:
                    if cor==True:
                        pipe.recon(method=rec, weeks=weeks, resample=resample, how='H', short=short, long=long, steps=hor, seasonal1=stat, seasonal2=stat*7)
                        print("Correction is on: ", str(cor))
                        pipe.outliers(0, 0.1, correction=cor)
                    else:
                        pipe.recon(method=rec, weeks=weeks, resample=resample, how='H', short=short, long=long, steps=hor, seasonal1=stat, seasonal2=stat*7)
                        print("Correction is on: ", str(cor))

                    for mod in models:
                        print("Model is: ", str(mod).split('(')[0])
                        pipe.predict(mod, stat=stat, horizon=hor)

                        np.savetxt("Results/" + str(rec)+"-"+str(short)+"-"+str(long)+"-"+"Correction_"+str(cor)+"-"+str(mod).split('(')[0]+"-"+str(stat)+"-"+str(hor)+".csv", pipe._y_forecast, delimiter=';')
    else:
        for cor in corr:
            if cor==True:
                pipe.recon(rec, weeks, resample=resample, how='H', short=None, long=None, steps=hor, seasonal1=stat, seasonal2=stat*7)
                print("Correction is on: ", str(cor))
                pipe.outliers(0, 0.1, correction=cor)
            else:
                pipe.recon(rec, weeks, resample=resample, how='H', short=None, long=None, steps=hor, seasonal1=stat, seasonal2=stat*7)
                print("Correction is on: ", str(cor))

                for mod in models:
                    print("Model is: ", str(mod).split('(')[0])
                    pipe.predict(mod, stat=stat, horizon=hor)

                    np.savetxt("Results/" + str(rec)+"-"+"Correction_"+str(cor)+"-"+str(mod).split('(')[0]+"-"+str(stat)+"-"+str(hor)+".csv", pipe._y_forecast, delimiter=';')

# path = 'C:\\Users\\Propietario\Documents\GitHub\TFM\Results'
# for file in os.listdir(path):
#     res = pd.read_csv(path +'\\'+ file, sep=';')