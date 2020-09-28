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

# recon_meths = ['mean', 'median', 'hybrid']
recon_meths = ['hybrid']
long_meths = ['mean', 'median']
short_meths = ['HW']
# short_meths = ['KNN', 'RF', 'SVR']

corr = [True, False]
nn = [False, True]
models = [KNeighborsRegressor(), RandomForestRegressor(), SVR()]


stat=96
hor=96

pipe = Pipeline(df_flow)
pipe.wrangle()

for rec in recon_meths:
    print("Recon meth is: ", str(rec))
    if rec=='hybrid':
        for short in short_meths:
            print("Short recon meth is: ", str(short))
            for long in long_meths:
                print("Long recon meth is: ", str(long))
                pipe.recon(method=rec, weeks=6, short=short, long=long)
                
                for cor in corr:
                    if cor==True:
                        print("Correction is on: ", str(cor))
                        pipe.outliers(0, 0.1, correction=cor)
                    else:
                        print("Correction is on: ", str(cor))
                    
                    for neural in nn:
                        if neural==True:
                            print("Model is neural network")
                            pipe.predict(stat=stat, horizon=hor)
                            np.savetxt("Results/" + str(rec)+"-"+str(short)+"-"+str(long)+"-"+str(cor)+"-"+"Neural"+"-"+str(stat)+"-"+str(hor)+".csv", pipe._y_forecast, delimiter=';')
                        else:
                            for mod in models:
                                print("Model is: ", str(mod).split('(')[0])
                                pipe.predict(mod, stat=stat, horizon=hor)
                                
                                np.savetxt("Results/" + str(rec)+"-"+str(short)+"-"+str(long)+"-"+str(cor)+"-"+str(mod).split('(')[0]+"-"+str(stat)+"-"+str(hor)+".csv", pipe._y_forecast, delimiter=';')
                
    else:
        pipe.recon(rec, 6)

    for cor in corr:
        if cor==True:
            print("Correction is on: ", str(cor))
            pipe.outliers(0, 0.1, correction=cor)
        else:
            print("Correction is on: ", str(cor))
        
        for neural in nn:
            if neural==True:
                print("Model is neural network")
                pipe.predict(stat=stat, horizon=hor)
                
                np.savetxt("Results/" + str(rec)+"-"+str(cor)+"-"+"Neural"+"-"+str(stat)+"-"+str(hor)+".csv", pipe._y_forecast, delimiter=';')
            else:
                for mod in models:
                    print("Model is: ", str(mod).split('(')[0])
                    pipe.predict(mod, stat=stat, horizon=hor)
                    
                    np.savetxt("Results/" + str(rec)+"-"+str(cor)+"-"+str(mod).split('(')[0]+"-"+str(stat)+"-"+str(hor)+".csv", pipe._y_forecast, delimiter=';')

path = 'C:\\Users\\Propietario\Documents\GitHub\TFM\Data'
for file in os.listdir(path):
    res = pd.read_csv(path +'\\'+ file, sep=';')