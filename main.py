# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:30:18 2020

@author: Propietario
"""

from waTS import Pipeline
import pandas as pd
import numpy as np
import metrics
from ypstruct import structure
import time as tm

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import os

df_flow = pd.read_csv("Data/flow.txt", parse_dates=[3], sep=';', header = None, skiprows=0).drop([0, 1, 2], axis=1)
recon_meths = ['mean', 'median', 'hybrid']
long_meths = ['mean', 'median']
short_meths = ['ARIMA', 'HW', 'KNN', 'RF', 'SVR']
corr = [True, False]
nn = [False, True]

models = [KNeighborsRegressor(n_neighbors=22,
                                      weights='uniform',
                                      algorithm='auto',
                                      leaf_size=73.76,
                                      p=1.09,
                                      n_jobs=-1), 
          RandomForestRegressor(n_estimators=380,
                                max_depth=62, 
                                min_samples_split=13, 
                                min_samples_leaf=9), 
          SVR(), 
          DecisionTreeRegressor(criterion='mse',
                                      splitter='best',
                                      max_depth=40,
                                      min_samples_split=34,
                                      min_samples_leaf=0.014596969), 
          GradientBoostingRegressor(loss='ls',
                        learning_rate=0.3,
                        n_estimators=58,
                        criterion='mse',
                        min_samples_split=16,
                        min_samples_leaf=0.0187,
                        max_depth=32),
          'ANN']

resample = False
stat=96
hor=96
weeks = 6

pipe = Pipeline(df_flow)
pipe.wrangle(plot=0)

time = structure()
time.imputation = None
time.correction = None
time.forecast = None

df_time = pd.DataFrame({"Model": [],
                "Imputation": [],
                "Correction": [],
                "Forecast": []})

for rec in recon_meths:
    print("Recon meth is: ", str(rec))
    if rec=='hybrid':
        for short in short_meths:
            print("Short recon meth is: ", str(short))
            for long in long_meths:
                print("Long recon meth is: ", str(long))
                for cor in corr:
                    if cor==True:
                        start_time = tm.time()
                        pipe.recon(method=rec, weeks=weeks, resample=resample, how='H', short=short, long=long, steps=hor, seasonal1=stat, seasonal2=stat*7)
                        time.imputation = tm.time()-start_time
                        print("Correction is on: ", str(cor))
                        start_time = tm.time()
                        pipe.outliers(0, 0.05, correction=cor)
                        time.correction = tm.time()-start_time
                    else:
                        start_time = tm.time()
                        pipe.recon(method=rec, weeks=weeks, resample=resample, how='H', short=short, long=long, steps=hor, seasonal1=stat, seasonal2=stat*7)
                        print("Correction is on: ", str(cor))
                        time.imputation = tm.time()-start_time
                        time.correction = 0
                    for mod in models:
                        start_time = tm.time()
                        print("Model is: ", str(mod).split('(')[0])
                        pipe.predict(mod, stat=stat, horizon=hor)
                        time.forecast = tm.time()-start_time
                        
                        df_time = df_time.append({"Model": str(rec)+"-"+str(short)+"-"+str(long)+"-"+"Correction_"+str(cor)+"-"+str(mod).split('(')[0]+"-"+str(stat)+"-"+str(hor),
                                                  "Imputation": time.imputation,
                                                  "Correction": time.correction,
                                                  "Forecast": time.forecast}, ignore_index=True)
                        np.savetxt("./Predictions_v2/" + str(rec)+"-"+str(short)+"-"+str(long)+"-"+"Correction_"+str(cor)+"-"+str(mod).split('(')[0]+"-"+str(stat)+"-"+str(hor)+".csv", pipe._y_forecast, delimiter=';')
    else:
        for cor in corr:
            if cor==True:
                start_time = tm.time()
                pipe.recon(rec, weeks, resample=resample, how='H', short=None, long=None, steps=hor, seasonal1=stat, seasonal2=stat*7)
                time.imputation = tm.time()-start_time
                print("Correction is on: ", str(cor))
                start_time = tm.time()
                pipe.outliers(0, 0.05, correction=cor)
                time.correction = tm.time()-start_time
            else:
                start_time = tm.time()
                pipe.recon(rec, weeks, resample=resample, how='H', short=None, long=None, steps=hor, seasonal1=stat, seasonal2=stat*7)
                print("Correction is on: ", str(cor))
                time.correction = 0

            for mod in models:
                start_time = tm.time()
                print("Model is: ", str(mod).split('(')[0])
                pipe.predict(mod, stat=stat, horizon=hor)
                time.forecast = tm.time()-start_time
                df_time = df_time.append({"Model": str(rec)+"-"+"Correction_"+str(cor)+"-"+str(mod).split('(')[0]+"-"+str(stat)+"-"+str(hor),
                                              "Imputation": time.imputation,
                                              "Correction": time.correction,
                                              "Forecast": time.forecast}, ignore_index=True)
                np.savetxt("./Predictions_v2/" + str(rec)+"-"+"Correction_"+str(cor)+"-"+str(mod).split('(')[0]+"-"+str(stat)+"-"+str(hor)+".csv", pipe._y_forecast, delimiter=';')