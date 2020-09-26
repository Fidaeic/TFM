# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:30:18 2020

@author: Propietario
"""

from waTS import *
import pandas as pd

df_flow = pd.read_csv("Data/flow.txt", parse_dates=[3], sep=';', header = None, skiprows=0).drop([0, 1, 2], axis=1)

recon_meths = ['mean', 'median', 'hybrid']
corr = [True, False]
nn = [False, True]
models = [KNeighborsRegressor(), RandomForestRegressor(), SVR()]


pipe = Pipeline(df_flow)
pipe.wrangle()

for rec in recon_meths:
    pipe.recon(rec, 6)

    for cor in corr:
        pipe.outliers(0, 0.1, correction=cor)
        
        for neural in nn:
            if neural==True:
                pipe.predict()
            else:
                for mod in models:
                    pipe.predict(mod, stat=672, horizon=672)