# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:15:43 2020

@author: Fidae El Morer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.svm import SVR
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time
#import xgboost as xgb

#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
#from keras import metrics

# =============================================================================
# DEFINICIÓN DE FUNCIONES
# =============================================================================

    # =============================================================================
    # Función para la manipulación de datos. Se eliminan duplicados y se introducen
    # registros vacíos en zonas donde no había datos
    # =============================================================================

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

def graficado_dia(df, dia, mes, anio):
    plt.figure(figsize=(20, 10))
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Flow", fontsize=18)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.plot(df[f"{anio}-{mes}-{dia}"])

def reconstruccion(df, metodo='mediana', semanas=6):
    dataframe = df.copy()
    #Este método únicamente tendrá en cuenta la mediana de los días que se le haya introducido
    if metodo == 'mediana':
        for n in nul:
            valores = []
            for k in range(semanas):
                valores.append(dataframe.loc[n-pd.Timedelta(value=(k+1)*7, unit='D')])
            dataframe.loc[n] = np.median(valores)

    #Este método hace lo mismo que el anterior, solo que con la media
    if metodo == 'media':
        for n in nul:
            valores = []
            for k in range(semanas):
                valores.append(dataframe.loc[n-pd.Timedelta(value=(k+1)*7, unit='D')])
            dataframe.loc[n] = np.mean(valores)
    return dataframe

def forecast(df, model, estac=96, horizonte=96):
    start_time = time.time()
    dataframe = pd.DataFrame()
    for i in range(estac, 0, -1):
        dataframe['t-'+str(i)] = df.Flow.shift(i, fill_value=np.nan)
    dataframe['t'] = df.Flow.values
    ind_0 = dataframe[dataframe[f't-{estac}'].isna() == False].index[0]
    aux = dataframe.loc[ind_0:]
    X = aux.iloc[:-horizonte, :-1]
    y = aux.iloc[:-horizonte, -1]
    y_for_test = aux.iloc[-horizonte:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print('Precisión entrenamiento: ', r2_score(y_train, y_pred_train)*100, "%")
    print('Precisión validación: ', r2_score(y_test, y_pred_test)*100, "%")

    X_for = X.iloc[-1, 1:]
    X_for['t'] = y.iloc[-1]
    X_for = np.array(X_for).reshape(1, estac)
    df_for = pd.DataFrame(X_for)
    df_for[estac] = model.predict(np.array(df_for.iloc[0]).reshape(1, estac))

    for i in range(1, horizonte):
        df_for = df_for.append(df_for.iloc[i-1, :].shift(-1))
        df_for.iloc[i, estac] = model.predict(np.array(df_for.iloc[i, :-1]).reshape(1, estac))

    y_forecast = df_for[estac].values
    print('Precisión predicción: ', r2_score(y_for_test, y_forecast)*100, "%")

    elapsed_time = time.time()-start_time
    print("Tiempo transcurrido: ", elapsed_time, " segundos")

    plt.figure(figsize=(20, 10))
    xlabels = y_for_test.index
    plt.plot(y_for_test.values, color='black')
    plt.plot(y_forecast, color='blue')
    plt.xticks(labels=xlabels, rotation=40, ha='right')
    plt.ylabel("Consumo de agua medido en m3", fontsize=16)
    plt.legend(["Consumo real", "Consumo predicho"], fontsize=14)
    return y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time



def forecast_NN(df, nodes, estac=96, horizonte=96, est=96, lay=3, init='normal', act='relu', opt='adam'):
    
    global lay
    
    def NN_model():
        #Creación del modelo
        model = Sequential()
        for _ in range(lay):
            model.add(Dense(nodes, input_dim=est, kernel_initializer=init, activation=act))
        model.add(Dense(1, kernel_initializer=init))
        # Compile model
        model.compile(optimizer=opt, loss='mse')
        return model

    start_time = time.time()
    dataframe = pd.DataFrame()
    for i in range(estac, 0, -1):
        dataframe['t-'+str(i)] = df.Flow.shift(i, fill_value=np.nan)
    dataframe['t'] = df.Flow.values
    ind_0 = dataframe[dataframe[f't-{estac}'].isna() == False].index[0]
    aux = dataframe.loc[ind_0:]
    X = aux.iloc[:-horizonte, :-1]
    y = aux.iloc[:-horizonte, -1]
    y_for_test = aux.iloc[-horizonte:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    estimator = KerasRegressor(build_fn=NN_model, nb_epoch=100, batch_size=32, verbose=False)
    
    estimator.fit(X_train, y_train)
    y_pred_train = estimator.predict(X_train)
    y_pred_test = estimator.predict(X_test)

    print('Precisión entrenamiento: ', r2_score(y_train, y_pred_train)*100, "%")
    print('Precisión validación: ', r2_score(y_test, y_pred_test)*100, "%")

    X_for = X.iloc[-1, 1:]
    X_for['t'] = y.iloc[-1]
    X_for = np.array(X_for).reshape(1, estac)
    df_for = pd.DataFrame(X_for)
    df_for[estac] = estimator.predict(np.array(df_for.iloc[0]).reshape(1, estac))

    for i in range(1, horizonte):
        df_for = df_for.append(df_for.iloc[i-1, :].shift(-1))
        df_for.iloc[i, estac] = estimator.predict(np.array(df_for.iloc[i, :-1]).reshape(1, estac))

    y_forecast = df_for[estac].values

    print('Precisión predicción: ', r2_score(y_for_test, y_forecast)*100, "%")
    elapsed_time = time.time()-start_time
    print("Tiempo transcurrido: ", elapsed_time, " segundos")

    plt.figure(figsize=(20, 10))
    xlabels = y_for_test.index
    plt.plot(y_for_test.values, color='black')
    plt.plot(y_forecast, color='blue')
    #plt.xticks(labels=xlabels, rotation=40, ha='right')
    plt.ylabel("Consumo de agua medido en m3", fontsize=16)
    plt.legend(["Consumo real", "Consumo predicho"], fontsize=14)
    return(y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time, estimator)

# =============================================================================
# EJECUCIÓN DEL CÓDIGO
# =============================================================================
df_flow = pd.read_csv("Data/flow.csv", parse_dates=[3], sep=',', header = None).drop([0, 1, 2], axis=1)
df_flow, almac, nul, bloques = data_wrangler(df_flow)
df_mediana = reconstruccion(df_flow)
#df_media = reconstruccion(df_flow, "media")
#y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time = forecast(df_mediana, SVR())
y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time, est = forecast_NN(df_mediana, 100, lay=15)
