# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:15:43 2020

@author: Fidae El Morer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time
#import xgboost as xgb
'''
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
#from keras import metrics
'''
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

def forecast(df, model, estac=96, horizonte=96, prt=1):
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
    if prt==1:
        plt.figure(figsize=(20, 10))
        xlabels = y_for_test.index
        plt.plot(y_for_test.values, color='black')
        plt.plot(y_forecast, color='blue')
        #plt.xticks(labels=xlabels, rotation=40, ha='right')
        plt.ylabel("Consumo de agua medido en m3", fontsize=16)
        plt.legend(["Consumo real", "Consumo predicho"], fontsize=14)
    return y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time

def forecast_NN(df, nodes, epochs, estac=96, horizonte=96, est=96, lay=3, init='normal', act='relu', opt='adam', prt=1):
    start_time = time.time()

    model = Sequential()
    
    for _ in range(lay):
        model.add(Dense(nodes, input_dim=est, kernel_initializer=init, activation=act))
        
    model.add(Dense(1, kernel_initializer=init))
    # Compile model
    model.compile(optimizer=opt, loss='mse')    

    
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

    # estimator = KerasRegressor(build_fn=NN_model, nb_epoch=100, batch_size=32, verbose=False)
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
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
        df_for.iloc[i, estac] = model.predict(np.array(df_for.iloc[i, :-1]).reshape(1, estac))[0]

    y_forecast = df_for[estac].values

    print('Precisión predicción: ', r2_score(y_for_test, y_forecast)*100, "%")
    elapsed_time = time.time()-start_time
    print("Tiempo transcurrido: ", elapsed_time, " segundos")
    if prt==1:
        plt.figure(figsize=(20, 10))
        xlabels = y_for_test.index
        plt.plot(y_for_test.values, color='black')
        plt.plot(y_forecast, color='blue')
        #plt.xticks(labels=xlabels, rotation=40, ha='right')
        plt.ylabel("Consumo de agua medido en m3", fontsize=16)
        plt.legend(["Consumo real", "Consumo predicho"], fontsize=14)
    return(y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time, model)

def nn_performance(nodes, epochs, layers):
    df_predicciones = pd.DataFrame({"Nodos":[],
                                   "Capas":[],
                                   "Epocas":[],
                                   "Predicción":[],
                                   "Precisión":[],
                                   "Tiempo":[]})

    for n in nodes:
        for e in epochs:
            for l in layers:
                for k in range(20):
                
                    print("Iteración ", k)
                    print("Nodos ", n)
                    print("Capas ", l)
                    print("Épocas ", e)
                
                    y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time, mod = forecast_NN(df=df_mediana, nodes=n, epochs=e, lay=l, prt=0)
                    
                    df_predicciones = df_predicciones.append({"Nodos":n,
                                            "Capas":l,
                                            "Epocas":e,
                                            "Predicción":y_forecast,
                                            "Precisión":r2_score(y_for_test, y_forecast)*100,
                                            "Tiempo":elapsed_time}, ignore_index=True)
                    df_predicciones.to_csv("Pred_NN.csv")


    return df_predicciones

def knn_performance(neigh, weights, leaf_size, est):
    df_predicciones = pd.DataFrame({"Vecinos":[],
                                   "Pesos":[],
                                   "Tamaño hoja":[],
                                   "Predicción":[],
                                   "Precisión":[],
                                   "Tiempo":[]})

    for n in neigh:
        for w in weights:
            for l in leaf_size:
                print("Vecinos ", n)
                print("Pesos ", w)
                print("Tamaño hoja ", l)
            
                y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time = forecast(df_mediana, KNeighborsRegressor(n_neighbors=n, weights=w, leaf_size=l), estac=est, prt=0)
                
                df_predicciones = df_predicciones.append({"Vecinos":n,
                                        "Pesos":w,
                                        "Tamaño hoja":l,
                                        "Predicción":y_forecast,
                                        "Precisión":r2_score(y_for_test, y_forecast)*100,
                                        "Tiempo":elapsed_time}, ignore_index=True)
                df_predicciones.to_csv("Pred_mediana_KNN"+str(est)+".csv", sep=';')


    return df_predicciones
# =============================================================================
# EJECUCIÓN DEL CÓDIGO
# =============================================================================
df_flow = pd.read_csv("Data/flow.csv", parse_dates=[3], sep=',', header = None).drop([0, 1, 2], axis=1)
df_flow, almac, nul, bloques = data_wrangler(df_flow)
df_mediana = reconstruccion(df_flow)
#df_media = reconstruccion(df_flow, "media")
'''
y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time = forecast(df_mediana, SVR(), estac=672)

y_pred_train_rf_1, y_pred_test_rf_1, df_for_rf_1, y_for_test_rf_1, y_forecast_rf_1, elapsed_time_rf_1 = forecast(df_mediana, RandomForestRegressor(), estac=672)
y_pred_train_rf_2, y_pred_test_rf_2, df_for_rf_2, y_for_test_rf_2, y_forecast_rf_2, elapsed_time_rf_2 = forecast(df_mediana, RandomForestRegressor(), estac=96)
y_pred_train_SVR_2, y_pred_test_SVR_2, df_for_SVR_2, y_for_test_SVR_2, y_forecast_SVR_2, elapsed_time_SVR_2 = forecast(df_mediana, SVR(), estac=96)
y_pred_train_KNN_1, y_pred_test_KNN_1, df_for_KNN_1, y_for_test_KNN_1, y_forecast_KNN_1, elapsed_time_KNN_1 = forecast(df_mediana, KNeighborsRegressor(), estac=672)
y_pred_train_KNN_2, y_pred_test_KNN_2, df_for_KNN_2, y_for_test_KNN_2, y_forecast_KNN_2, elapsed_time_KNN_2 = forecast(df_mediana, KNeighborsRegressor(), estac=96)

y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time, mod = forecast_NN(df=df_mediana, nodes=20, epochs=20, lay=5, estac=672, est=672, horizonte=96)


nodes = [10, 30, 50]
epochs = [10, 100]
layers = [3, 5, 7]

df_pred = nn_performance(nodes, epochs, layers)

#El tamaño de hoja no influye ne los resultados y los pesos basados en la distancia funcionan peor, así que solo se usarán los uniformes
vecinos = range(5, 20, 2)
pesos = ['uniform']
tam_hoja = [30]

df_pred_knn = knn_performance(vecinos, pesos, tam_hoja, est=672)
'''

from tbats import TBATS

estimator = TBATS(seasonal_periods=[96, 672])

fitted_model = estimator.fit(df_mediana)

y_forecasted = fitted_model.forecast(steps=96)
