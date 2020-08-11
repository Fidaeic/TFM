# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:32:20 2020

@author: Propietario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import svd, eig
from MSPC_Caudales import PCA
from statsmodels.multivariate.pca import PCA
from sklearn.cluster import KMeans
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.svm import SVR
#from sklearn.linear_model import LinearRegression


def data_wrangler(df):
    df.columns = ["DateTime", "Flow"]
    df = df.drop_duplicates(subset="DateTime", keep='first')
    df = df.sort_values(by="DateTime").reset_index()
    r = pd.date_range(start=df_flow.DateTime.min(), end=df_flow.DateTime.max(), freq='15min')
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
    for i in range(1, len(alm)-1):
        chunks.append(df.loc[alm[i][1]:alm[i+1][0]].index)
    chunks.append(df.loc[alm[-1][1]:nulos[-1]].index)

    #Dibujamos en un gráfico las posiciones en las que hay datos vacíos
    plt.figure(figsize=(30, 10))
    plt.xlabel("Time", fontsize=18)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.plot(df.isna())
    return df, alm, nulos, chunks



df_flow = pd.read_csv("Data/flow.csv", parse_dates=[3], sep=',', header = None).drop([0, 1, 2], axis=1)
df_flow, almac, nul, bloques = data_wrangler(df_flow)
df_mediana = reconstruccion(df_flow)


df_prueba = df_mediana.loc[:nul[0]]


def embedding(df, M):
    
    N = df.shape[0]
    K = N-M+1
    
    mat = np.zeros((M,1))
    
    for i in range(K):
        z = df.iloc[i:i+M]
        
        mat = np.append(mat, z, axis=1)
    
    C_x = (1/K)*np.dot(mat, np.transpose(mat))
    
    
    return mat, C_x

m, C_x = embedding(df_prueba, 100)
m.shape

E = eig(C_x)

festivos = ["2018-01-01","2018-01-06","2018-03-19","2018-03-30","2018-04-02","2018-04-09",
           "2018-05-01","2018-06-25","2018-08-15","2018-10-09", "2018-10-12","2018-11-01",
           "2018-12-06","2018-12-08","2018-12-25",
           "2019-01-01","2019-01-06","2019-03-19","2019-04-18","2019-04-19","2019-04-22","2019-04-29",
           "2019-05-01","2019-06-24","2019-08-15","2019-10-09", "2019-10-12","2019-11-01","2019-12-06",
           "2019-12-08","2019-12-25","2020-01-01","2020-01-06"]




data = matrizado(festivos, df_flow)
d = data[:-1,:-6]
d=np.append(d,np.zeros((658,1)), axis=1)

km = KMeans(30)
km.fit(d)


d[:,-1] = km.labels_

plt.plot(d[:,:-1][d[:,-1]==1])
