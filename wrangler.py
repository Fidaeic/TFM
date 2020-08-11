# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:27:16 2020

@author: Propietario
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

def data_wrangler(df):
    df.columns = ["DateTime", "Flow"]
    df = df.drop_duplicates(subset="DateTime", keep='first')
    df = df.sort_values(by="DateTime").reset_index()
    r = pd.date_range(start=df.DateTime.min(), end=df.DateTime.max(), freq='15min')
    df = df.set_index('DateTime')
    df = df.reindex(r)
    df.drop(labels="index", axis=1, inplace=True)
    nulos = np.array(df[df["Flow"].isna()].index)

    #Almacenamos los datos nulos
    alm = []
    for i in range(1, len(nulos)):
        delta = nulos[i]- nulos[i-1]
        delta = delta.astype('timedelta64[m]')
        if delta != np.timedelta64(15, 'm'):
            alm.append((nulos[i-1], nulos[i]))

    #Conseguimos los bloques de datos nulos
    chunks = []
    chunks.append(df.loc[nulos[0]:alm[0][0]].index)
    for i in range(0, len(alm)-1):
        chunks.append(df.loc[alm[i][1]:alm[i+1][0]].index)
    chunks.append(df.loc[alm[-1][1]:nulos[-1]].index)

    #Dibujamos en un gráfico las posiciones en las que hay datos vacíos
    plt.figure(figsize=(30, 10))
    plt.xlabel("Time", fontsize=18)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.plot(df.isna())
    return df, alm, nulos, chunks

def matrizado(fest, df_flow):
    
    days = len(df_flow.groupby(df_flow.index.floor('d')).size())
    
    data = np.zeros((days,102))
    
    for d, t in zip(range(days), range(0, df_flow.size, 96)):
        datos[d,:96] = df_flow.iloc[t:t+96].values.reshape(96)
        datos[d,97] = int(df_flow.index[t].dayofweek+1)
        datos[d,98] = int(df_flow.index[t].day)
        datos[d,99] = int(df_flow.index[t].month)
        datos[d,100] = int(df_flow.index[t].year)
        
        for f in fest:
            if df_flow.index[t]==pd.to_datetime(f):
                datos[d,101] = 7
    
    data = data[~np.isnan(datos).any(axis=1)]
    
    return data

def shifting(df, estac, horizon, tt_split=True, t_size=0.3):
    dataframe = pd.DataFrame()
    
    for i in range(estac, 0, -1):
        dataframe['t-'+str(i)] = df.Flow.shift(i, fill_value=np.nan)
        
    dataframe['t'] = df.Flow.values
    ind_0 = dataframe[dataframe[f't-{estac}'].isna() == False].index[0]
    aux = dataframe.loc[ind_0:]
    X = aux.iloc[:-horizon, :-1]
    y = aux.iloc[:-horizon, -1]
    y_for_test = aux.iloc[-horizon:, -1]
    
    if tt_split == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, shuffle=False)
        return X_train, X_test, y_train, y_test, X, y, y_for_test
    else:
        return X, y, y_for_test
    
def graficado_dia(df, dia, mes, anio):
    plt.figure(figsize=(20, 10))
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Flow", fontsize=18)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.plot(df[f"{anio}-{mes}-{dia}"])