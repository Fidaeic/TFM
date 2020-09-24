# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 11:07:28 2020

@author: Propietario
"""

import pandas as pd
import numpy as np
import wrangler
from scipy.stats import norm
import recon
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt

def clusters(mat, n_clusters):
    '''
    Parameters
    ----------
    mat : numpy array
        Matriz en la que las filas son los días y las columnas son los instantes del día, obtenida con wrangler.matrizado.

    Returns
    -------
    mat : numpy array
        Matriz que tiene el clúster al que pertenece cada día en la última columna.
    median : numpy array
        Matriz con las medianas de cada clúster.

    '''
    clus = KMeans(n_clusters=n_clusters).fit(mat)
    df = pd.DataFrame(mat)
    df['labels'] = clus.labels_
    mat = np.array(df)

    median = np.zeros((len(np.unique(clus.labels_)), 96))
    for i in np.unique(clus.labels_):
        a = mat[mat[:,-1]==i]
        for j in range(96):
            median[i, j] = np.median(a[:, j])
    return mat, median, clus.inertia_

def Qn(x):
    '''
    Parameters
    ----------
    x : numpy array or list
        DESCRIPTION.

    Returns
    -------
    qn : float
        Qn estimator.
    '''

    n = len(x)
    h = (n//2)+1
    k = (h*(h-1))//2

    ser = []

    for i in range(len(x)):
        for j in range(1, len(x)):
            if j>i:
                ser.append(abs(x[i]-x[j]))
    ser = sorted(ser)
    qn = 2.2219*ser[k]
    return qn

def outlier_region(df, n_clusters):
    '''
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.

    Returns
    -------
    pr : TYPE
        DESCRIPTION.
    '''
    mat = wrangler.matrizado(0, df)
    cl, med, inertia = clusters(mat, n_clusters)
    qn = []

    for i in range(med.shape[0]):
        a = cl[cl[:,-1]==i][:,:96]
        b = np.reshape(a,a.size)
        qn.append(Qn(b))

    upper_90 = []
    lower_90 = []

    upper_95 = []
    lower_95 = []

    upper_99 = []
    lower_99 = []

    for row in cl:
        a = row[:96]
        cluster = int(row[102])
        m = med[cluster, :]
        upper_90.append([x + norm.ppf(1-0.1/2)*qn[cluster] for x in m])
        lower_90.append([x - norm.ppf(1-0.1/2)*qn[cluster] for x in m])

        upper_95.append([x + norm.ppf(1-0.05/2)*qn[cluster] for x in m])
        lower_95.append([x - norm.ppf(1-0.05/2)*qn[cluster] for x in m])

        upper_99.append([x + norm.ppf(1-0.01/2)*qn[cluster] for x in m])
        lower_99.append([x - norm.ppf(1-0.01/2)*qn[cluster] for x in m])

    upper_90 = np.array(upper_90)
    upper_90 = upper_90.reshape(upper_90.size)

    lower_90 = np.array(lower_90)
    lower_90 = lower_90.reshape(lower_90.size)
    
    upper_95 = np.array(upper_95)
    upper_95 = upper_95.reshape(upper_95.size)

    lower_95 = np.array(lower_95)
    lower_95 = lower_95.reshape(lower_95.size)
    
    upper_99 = np.array(upper_99)
    upper_99 = upper_99.reshape(upper_99.size)

    lower_99 = np.array(lower_99)
    lower_99 = lower_99.reshape(lower_99.size)

    medianas = []
    for row in cl:
        medianas.append(med[int(row[102])])
        
    medianas = np.array(medianas)
    medianas = medianas.reshape(medianas.size)
    
    dataframe = pd.DataFrame()
    
    dataframe["Flow"] = df.Flow
    dataframe["Upper_90"] = upper_90
    dataframe["Lower_90"] = lower_90

    dataframe["Upper_95"] = upper_95
    dataframe["Lower_95"] = lower_95

    dataframe["Upper_99"] = upper_99
    dataframe["Lower_99"] = lower_99

    dataframe["Median"] = medianas
    return dataframe

def graficado(dia, mes, ano, df, alpha):

    fecha = str(ano)+"-"+str(mes)+"-"+str(dia)
    plt.figure(figsize=(16,10))
    plt.grid(alpha=0.5)
    plt.plot(df.loc[fecha, "Median"], color='black', linestyle='dashed')
    plt.plot(df.loc[fecha, "Flow"])
    
    plt.title(f"Consumption and outlier region on day {ano}-{mes}-{dia}", fontsize=20)
    plt.ylabel("Water consumption (m3/h)", fontsize=16)
    plt.xticks(rotation=45)

    if alpha == 0.1:
        plt.plot(df.loc[fecha, "Upper_90"], color='red', linestyle='dashed')
        plt.plot(df.loc[fecha, "Lower_90"], color='red', linestyle='dashed')
    elif alpha == 0.05:
        plt.plot(df.loc[fecha, "Upper_95"], color='red', linestyle='dashed')
        plt.plot(df.loc[fecha, "Lower_95"], color='red', linestyle='dashed')
    elif alpha == 0.01:
        plt.plot(df.loc[fecha, "Upper_99"], color='red', linestyle='dashed')
        plt.plot(df.loc[fecha, "Lower_99"], color='red', linestyle='dashed')
    else:
        plt.plot(df.loc[fecha, "Upper_90"], color='red', linestyle='dashed')
        plt.plot(df.loc[fecha, "Lower_90"], color='red', linestyle='dashed')
        plt.plot(df.loc[fecha, "Upper_95"], color='red', linestyle='dashed')
        plt.plot(df.loc[fecha, "Lower_95"], color='red', linestyle='dashed')  
        plt.plot(df.loc[fecha, "Upper_99"], color='red', linestyle='dashed')
        plt.plot(df.loc[fecha, "Lower_99"], color='red', linestyle='dashed')
        
        plt.legend(["Median of the cluster", "Water consumption", "Outlier region"], fontsize=16)

def corrected(df_outliers, alpha):
    correct = np.zeros(df_outliers.shape[0])
    i = 0
    if alpha == 0.1:
        for index, row in df_outliers.iterrows():
            if row["Flow"]>row["Upper_90"] or row["Flow"]<row["Lower_90"]:
                correct[i] = row["Median"]
            else:
                correct[i] = row["Flow"]
            i+=1
    elif alpha == 0.05:
        for index, row in df_outliers.iterrows():
            if row["Flow"]>row["Upper_95"] or row["Flow"]<row["Lower_95"]:
                correct[i] = row["Median"]
            else:
                correct[i] = row["Flow"]
            i+=1
    elif alpha == 0.01:
        for index, row in df_outliers.iterrows():
            if row["Flow"]>row["Upper_99"] or row["Flow"]<row["Lower_99"]:
                correct[i] = row["Median"]
            else:
                correct[i] = row["Flow"]
            i+=1
    df_correct = pd.DataFrame({"Flow": correct}, index=df_outliers.index)
    return df_correct

# df_flow = pd.read_csv("Data/flow.csv", parse_dates=[3], sep=',', header = None).drop([0, 1, 2], axis=1)
# df_flow, almac, nul, bloques = wrangler.data_wrangler(df_flow)
# # df_mediana, time = recon.recon_mean(df_flow, nul)
# dataf, time = recon.recon_median(df_flow, nul, weeks=4)
# dataf = dataf.iloc[:-4]


# df = outlier_region(dataf, 20)

# # for i in range(1, 31):
# alpha = 0.1
# for i in range(1, 31):
#     graficado(i, 10, 2018, df, 0)

# graficado(1, 11, 2018, df, 0)
# graficado(2, 11, 2018, df, 0)
# graficado(3, 11, 2018, df, 0)



# df_corr = corrected(df, 0.05)

#Diferencia entre serie corregida y no corregida
# plt.plot(df.loc['2018-11-1', 'Flow'])
# plt.plot(df_corr.loc['2018-11-1'])