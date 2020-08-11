# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:56:49 2020

@author: Fidae El Morer
"""
import pandas as pd
import numpy as np
import wrangler
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.model_selection import LeaveOneOut, KFold, train_test_split

from sklearn.metrics import r2_score
import time
#import xgboost as xgb
from scipy.stats import norm

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


# =============================================================================
# FORECASTING METHODS FOR WATER CONSUMPTION TIME SERIES
# =============================================================================
def forecast(df, model, estac=96, horizon=96, prt=1):
    '''
    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe with only one column called "Flow" and and a DateTime index
    model :  Sklearn objects
        Machine Learning methods present in the Sklearn library.
    estac : int, optional
        Stationality of the time series.The default is 96, considering flow values every 15 minutes during a day.
    horizon : int, optional
        Number of steps to be forecasted. The default is 96, considering flow values every 15 minutes during a day.
    prt : int, optional
        The values can be 0 or 1. If its value is 1, the forecasted values will be displayed 
        on a plot along with the actual consumption values. The default is 1.

    Returns
    -------
    y_pred_train : Pandas series
        Predicted values for the training set.
    y_pred_test : Pandas series
        Predicted values for the test set.
    y_for_test : Pandas series
        Actual values to test the forecasting power of the algorithm.
    y_forecast : TYPE
        Out of sample forecasted values.
    elapsed_time : float
        Elapsed time to perform the imputing method.

    '''
    start_time = time.time()
    
    X_train, X_test, y_train, y_test, X, y, y_for_test= wrangler.shifting(df, estac, horizon)

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

    for i in range(1, horizon):
        df_for = df_for.append(df_for.iloc[i-1, :].shift(-1))
        df_for.iloc[i, estac] = model.predict(np.array(df_for.iloc[i, :-1]).reshape(1, estac))

    y_forecast = df_for[estac].values
    print('Precisión predicción: ', r2_score(y_for_test, y_forecast)*100, "%")
    
    # Training confidence intervals
    res_train = y_pred_train-y_train
    mu_train = np.mean(res_train)
    sigma_train = np.std(res_train)    
    
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
    return y_pred_train, y_pred_test, y_for_test, y_forecast, elapsed_time

def forecast_NN(df, nodes, epochs, estac=96, horizon=96, lay=3, init='normal', act='relu', opt='adam', prt=1):
    '''
    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe with only one column called "Flow" and and a DateTime index
    nodes : int
        Number of nodes in each layer.
    epochs : int
        Number of epochs.
    estac : int, optional
        Stationality of the time series.The default is 96, considering flow values every 15 minutes during a day.
    horizon : TYPE, optional
        Number of steps to be forecasted. The default is 96, considering flow values every 15 minutes during a day.
    lay : int, optional
        Number of layers. The default is 3.
    init : string, optional
        Initialization method for the layers. The default is 'normal'.
    act : string, optional
        Activation method. The default is 'relu'.
    opt : string, optional
        Optimizer. The default is 'adam'.
    prt : int, optional
        The values can be 0 or 1. If its value is 1, the forecasted values will be displayed 
        on a plot along with the actual consumption values. The default is 1.

    Returns
    -------
    y_pred_train : Pandas series
        Predicted values for the training set.
    y_pred_test : Pandas series
        Predicted values for the test set.
    y_for_test : Pandas series
        Actual values to test the forecasting power of the algorithm.
    y_forecast : TYPE
        Out of sample forecasted values.
    elapsed_time : float
        Elapsed time to perform the imputing method.
    '''
    start_time = time.time()

    model = Sequential()
    
    for _ in range(lay):
        model.add(Dense(nodes, input_dim=estac, kernel_initializer=init, activation=act))
        
    model.add(Dense(1, kernel_initializer=init))
    # Compile model
    model.compile(optimizer=opt, loss='mse')    

    X_train, X_test, y_train, y_test, X, y, y_for_test= wrangler.shifting(df, estac, horizon)

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

    for i in range(1, horizon):
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
    return y_pred_train, y_pred_test, y_for_test, y_forecast, elapsed_time

def nn_performance(df, nodes, epochs, layers, it):
    
    df_predicciones = pd.DataFrame({"Nodos":[],
                                   "Capas":[],
                                   "Epocas":[],
                                   "Predicción":[],
                                   "Precisión":[],
                                   "Tiempo":[]})

    for n in nodes:
        for e in epochs:
            for l in layers:
                for k in range(it):
                
                    print("Iteración ", k)
                    print("Nodos ", n)
                    print("Capas ", l)
                    print("Épocas ", e)
                
                    y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time, mod = forecast_NN(df, nodes=n, epochs=e, lay=l, prt=0)
                    
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

