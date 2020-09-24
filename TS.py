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
import outies
from sklearn.pipeline import Pipeline
from ypstruct import structure
from scipy.stats import normaltest
# =============================================================================
# RECONSTRUCCIÓN
# =============================================================================
#Importación de serie y relleno de huecos con valores faltantes
df_flow = pd.read_csv("Data/flow.csv", parse_dates=[3], sep=',', header = None).drop([0, 1, 2], axis=1)
df_flow, almac, nul, bloques = wrangler.data_wrangler(df_flow)

#Imputación de valores faltantes y eliminación de los últimos 4 registros para tener días completos
dataf, time = recon.recon_hybrid(df_flow, bloques, 96, short='KNN', weeks=4)
dataf = dataf.iloc[:-4]

#Determinación de valores anómalos y gráficos
df = outies.outlier_region(dataf, n_clusters=7)

for i in range(1, 31):
    outies.graficado(i, 6, 2018, df, 0)

#Sustitución de los valores anómalos por la mediana del cluster

alpha = 0.05
df_corr = outies.corrected(df, alpha)

#Podemos ver la diferencia para un mismo día en el que hemos sustituido los valores anómalos por la mediana
plt.plot(df.loc['2018-10-30', 'Flow'])
plt.plot(df_corr.loc['2018-10-30'])
# =============================================================================
# OPTIMIZACIÓN
# =============================================================================

# Problem Definition
problem = structure()
problem.costfunc = prec
problem.nvar = 4
#n_estimators, max_depth, min samples split, min samples leaf
problem.varmin = [1, 1, 2, 1]
problem.varmax = [100,  100, 20, 20]

# GA Parameters
params = structure()
params.maxit = 10
params.npop =10
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1

# Run GA
ga.out = run(problem, params, dataf, 96, 96, model='RF')

# =============================================================================
# PREDICCIÓN DE MODELOS INDIVIDUALES
# =============================================================================
# y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time = forecast(df_mediana, SVR(), estac=672)

y_pred_train_rf_1, y_pred_test_rf_1, y_for_test_rf_1, y_forecast_rf_1, elapsed_time_rf_1 = pred.forecast(df_corr, RandomForestRegressor(), estac=672)
y_pred_train_rf_2, y_pred_test_rf_2, y_for_test_rf_2, y_forecast_rf_2, elapsed_time_rf_2 = pred.forecast(df_corr, RandomForestRegressor(n_estimators=74, max_depth=50, min_samples_split=9, min_samples_leaf=9), estac=96)
# y_pred_train_SVR_2, y_pred_test_SVR_2, df_for_SVR_2, y_for_test_SVR_2, y_forecast_SVR_2, elapsed_time_SVR_2 = forecast(df_mediana, SVR(), estac=96)
y_pred_train_KNN_1, y_pred_test_KNN_1, y_for_test_KNN_1, y_forecast_KNN_1, elapsed_time_KNN_1 = pred.forecast(df_corr, KNeighborsRegressor(), estac=672)
y_pred_train_KNN_2, y_pred_test_KNN_2, y_for_test_KNN_2, y_forecast_KNN_2, elapsed_time_KNN_2 = pred.forecast(df_corr, KNeighborsRegressor(), estac=96)
# y_pred_train, y_pred_test, df_for, y_for_test, y_forecast, elapsed_time, mod = forecast_NN(df=df_mediana, nodes=20, epochs=20, lay=5, estac=672, est=672, horizonte=96)

# res_train = y_pred_train_KNN_1-y_train
# res_test = y_pred_test_KNN_2-y_test
# res_for = y_forecast_KNN_2-y_for_test_KNN_2
# plt.plot(res_for)

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

iner = []
for i in range(5,50):
    cl, med, inertia = outies.clusters(mat, i)
    
    iner.append(inertia)
plt.plot(iner)

from sklearn.decomposition import PCA
pr = mat[:, :96]

pca = PCA()

pca.fit(pr)
plt.plot(pca.explained_variance_ratio_)

plt.scatter(pca.components_[1], pca.components_[2])

from statsmodels.tsa.seasonal import seasonal_decompose

df_corr_2018 = df_corr.loc['2018']
df_corr_2019 = df_corr.loc['2019']

plt.plot(df_corr_2018)
plt.plot(df_corr_2019)

result = seasonal_decompose(df_corr_2018.Flow, model='additive', period=672)
result.plot()

result = seasonal_decompose(df_corr_2019.Flow, model='additive', period=672)
result.plot()

print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)

plt.plot(df_corr_2018.loc['2018-06'])

