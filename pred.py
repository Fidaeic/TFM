# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:56:49 2020

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