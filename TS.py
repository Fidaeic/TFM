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
import recon
#import xgboost as xgb
import wrangler
import pred
import ga
from sklearn.pipeline import Pipeline

from scipy.stats import normaltest
# =============================================================================
# RECONSTRUCCIÓN
# =============================================================================
df_flow = pd.read_csv("Data/flow.csv", parse_dates=[3], sep=',', header = None).drop([0, 1, 2], axis=1)
df_flow, almac, nul, bloques = wrangler.data_wrangler(df_flow)
# df_mediana, time = recon.recon_mean(df_flow, nul)
dataf, time = recon.recon_hybrid(df_flow, bloques, 96, short='KNN', weeks=4)
# dataf_RF, time_RF, y_for_RF = recon.recon_hybrid(df_flow, bloques, 96, short='RF', weeks=4)
# dataf_SVR, time_SVR, y_for_SVR = recon.recon_hybrid(df_flow, bloques, 96, short='SVR', weeks=4)
# plt.plot(dataf.loc['2019-03-31'])
# plt.plot(dataf_RF.loc['2019-03-31'])
# plt.plot(dataf_SVR.loc['2019-03-31'])
# =============================================================================
# PREDICCIÓN DE MODELOS INDIVIDUALES
# =============================================================================
# y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time = forecast(df_mediana, SVR(), estac=672)

# y_pred_train_rf_1, y_pred_test_rf_1, df_for_rf_1, y_for_test_rf_1, y_forecast_rf_1, elapsed_time_rf_1 = forecast(df_mediana, RandomForestRegressor(), estac=672)
# y_pred_train_rf_2, y_pred_test_rf_2, df_for_rf_2, y_for_test_rf_2, y_forecast_rf_2, elapsed_time_rf_2 = forecast(df_mediana, RandomForestRegressor(), estac=96)
# y_pred_train_SVR_2, y_pred_test_SVR_2, df_for_SVR_2, y_for_test_SVR_2, y_forecast_SVR_2, elapsed_time_SVR_2 = forecast(df_mediana, SVR(), estac=96)
# y_pred_train_KNN_1, y_pred_test_KNN_1, df_for_KNN_1, y_for_test_KNN_1, y_forecast_KNN_1, elapsed_time_KNN_1 = forecast(df_mediana, KNeighborsRegressor(), estac=672)
y_pred_train_KNN_2, y_pred_test_KNN_2, y_for_test_KNN_2, y_forecast_KNN_2, elapsed_time_KNN_2 = pred.forecast(dataf, KNeighborsRegressor(), estac=96)
# y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time, mod = forecast_NN(df=df_mediana, nodes=20, epochs=20, lay=5, estac=672, est=672, horizonte=96)

# X_train, X_test, y_train, y_test, X, y, y_for_test= wrangler.shifting(dataf, 96, 96)

res_train = y_pred_train_KNN_2-y_train
res_test = y_pred_test_KNN_2-y_test
res_for = y_forecast_KNN_2-y_for_test_KNN_2
plt.plot(res_for)

# =============================================================================
# PERFORMANCE
# =============================================================================
# nodes = [10, 30, 50]
# epochs = [10, 100]
# layers = [3, 5, 7]

# df_pred = nn_performance(nodes, epochs, layers)

# #El tamaño de hoja no influye ne los resultados y los pesos basados en la distancia funcionan peor, así que solo se usarán los uniformes
# vecinos = range(5, 20, 2)
# pesos = ['uniform']
# tam_hoja = [30]

# df_pred_knn = knn_performance(vecinos, pesos, tam_hoja, est=672)

# =============================================================================
# PIPELINE
# =============================================================================


