# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:30:18 2020

@author: Propietario
"""

from waTS import *
import pandas as pd
import itertools
import numpy as np
import metrics
import math
from ypstruct import structure
import time as tm

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
                        np.savetxt("Results_final/" + str(rec)+"-"+str(short)+"-"+str(long)+"-"+"Correction_"+str(cor)+"-"+str(mod).split('(')[0]+"-"+str(stat)+"-"+str(hor)+".csv", pipe._y_forecast, delimiter=';')
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
                np.savetxt("Results_final/" + str(rec)+"-"+"Correction_"+str(cor)+"-"+str(mod).split('(')[0]+"-"+str(stat)+"-"+str(hor)+".csv", pipe._y_forecast, delimiter=';')


# =============================================================================
# RESULTADOS
# =============================================================================
y_real_96 = pipe.ts.tail(96).values.reshape(96)

np.savetxt("Y_real.csv", y_real_96)

path = 'C:\\Users\\Propietario\Documents\GitHub\TFM\Results_final'

results = pd.DataFrame({"Recon_method": [],
                        "Short_Recon_Method": [],
                        "Long_Recon_Method": [],
                        "Corrected": [],
                        "Forecasting_Model": [],
                        "Seasonality": [],
                        "Horizon": [],
                        "R2": [],
                        "SMAPE": [],
                        "MAE": [],
                        "RMSE": []})
for file in os.listdir(path):
    name = file.split("-")
    res = np.loadtxt(path +'\\'+ file, delimiter=';')

    s = metrics.smape(res, y_real_96)
    m = metrics.mae(res, y_real_96)
    r = metrics.r2(res, y_real_96)
    rmse = metrics.rmse(res, y_real_96)


    if len(name)==7:
        results = results.append({"Recon_method": name[0],
                            "Short_Recon_Method": name[1],
                            "Long_Recon_Method": name[2],
                            "Corrected": name[3],
                            "Forecasting_Model": name[4],
                            "Seasonality": name[5],
                            "Horizon": name[6],
                            "R2": r,
                            "SMAPE": s,
                            "MAE": m,
                            "RMSE": rmse}, ignore_index=True)
    else:
        results = results.append({"Recon_method": name[0],
                    "Short_Recon_Method": None,
                    "Long_Recon_Method": None,
                    "Corrected": name[1],
                    "Forecasting_Model": name[2],
                    "Seasonality": name[3],
                    "Horizon": name[4],
                    "R2": r,
                    "SMAPE": s,
                    "MAE": m,
                    "RMSE": rmse}, ignore_index=True)


from scipy.stats import probplot, normaltest, kruskal, levene, chisquare, wilcoxon

fig = plt.figure()
ax = fig.add_subplot(111)
probplot(results[results['Forecasting_Model']=='GradientBoostingRegressor']['RMSE'], plot=ax)

alpha = .01
k2, p = normaltest(results[results['Forecasting_Model']=='GradientBoostingRegressor']['RMSE'])

k, p = levene(results[results['Recon_method']=='median']['RMSE'], results[results['Recon_method']=='mean']['RMSE'])
k, p = kruskal(results[results['Recon_method']=='median']['RMSE'], results[results['Recon_method']=='mean']['RMSE'])
k, p = ttest_ind(results[results['Recon_method']=='median']['RMSE'], results[results['Recon_method']=='mean']['RMSE'])
k, p = wilcoxon(results[results['Recon_method']=='median']['RMSE'], results[results['Recon_method']=='mean']['RMSE'])

if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")



rec_mean = results[results['Recon_method']=='mean']['RMSE']
rec_median = results[results['Recon_method']=='median']['RMSE']
rec_hybrid = results[results['Recon_method']=='hybrid']['RMSE']

import seaborn as sns

results.to_csv("Results.csv")


df_time['Recon_method'] = df_time['Model'].apply(lambda x: x.split("-")[0])
df_time['Forecasting_Model'] = df_time['Model'].apply(lambda x: x.split("-")[-3])



df_time['Total'] = df_time['Imputation']+df_time['Correction']+df_time['Forecast']



