# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:27:07 2020

@author: Propietario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import waTS

from scipy.stats import probplot, normaltest, kruskal, levene, chisquare, wilcoxon

def boxplot(category, value, df):

    sns.set_style(style="ticks")
    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(x=value, y=category, data=df,
                whis=[0, 100], width=.6, color="grey")
    # Add in points to show each observation
    sns.stripplot(x=value, y=category, data=df,
                  size=4, color=".3", linewidth=0)
    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    #sns.despine(trim=True, left=True)

def stat_test(sample_a, sample_b, alpha, crit):
    if crit =='levene':
        s, p = levene(sample_a, sample_b)
    elif crit =='kruskal':
        s, p = kruskal(sample_a, sample_b)
    elif crit=='wilcoxon':
        s, p = wilcoxon(sample_a, sample_b)
    elif crit == 'student':
        s, p = ttest_ind(sample_a, sample_b)

    if p<alpha:
        print(f"Prueba de {crit}\n")
        print(f"El P-valor es {p}")
        print("Rechazo de la hipótesis nula. Existen diferencias significativas")
    else:
        print(f"Prueba de {crit}\n")
        print(f"El P-valor es {p}")
        print("Aceptación de la hipótesis nula. Las diferencias no son significativas")
    return p

def recon_rows(row):
    if len(row['Recon_aux']) ==5:
        value = row['Recon_aux'][0]
    else:
        value = row['Recon_aux'][0]+'-'+ row['Recon_aux'][1]+'-'+row['Recon_aux'][2]
    return value
# =============================================================================
# CARGA DE DATOS
# =============================================================================

df_time = pd.read_csv('./Results/Time.csv')
df_results = pd.read_csv('./Results/Results.csv')
y_real = np.loadtxt('./Results/Y_real.csv')

stat_test(df_results[df_results['Recon_method']=='hybrid']['RMSE'], 
          df_results[df_results['Recon_method']=='median']['RMSE'], alpha=0.05, crit='kruskal')

df_results['Recon'] = np.where(df_results['Short_Recon_Method'].isna(), df_results['Recon_method'], df_results['Recon_method']+ '-'+df_results['Short_Recon_Method']+'-'+df_results['Long_Recon_Method'])

boxplot(category = 'Forecasting_Model', value='RMSE', df=df_results)

# =============================================================================
# SIGNIFICACIÓN ESTADÍSTICA DE LA IMPUTACIÓN DE VALORES FALTANTES
# =============================================================================

df_p_valor_imp = pd.DataFrame({"Método A": [],
                  'Método B': [],
                  'P-valor':[]})

methods = list(df_results['Recon_method'].unique())
alpha = 0.05
for met_a in methods:
    for met_b in methods:
        sample_a = df_results[df_results['Recon_method']==met_a]['RMSE']
        sample_b = df_results[df_results['Recon_method']==met_b]['RMSE']
        print('**********************************************')
        print(f"PRUEBA DE HIPÓTESIS PARA LOS MÉTODOS {met_a} Y {met_b}")
        df_p_valor_imp = df_p_valor_imp.append({"Método A": met_a,
                  'Método B': met_b,
                  'P-valor': stat_test(sample_a, sample_b, crit='kruskal', alpha=alpha)}, ignore_index=True)
        

pivot_imp = df_p_valor_imp.pivot(index='Método A',
                 columns = 'Método B',
                 values= 'P-valor')

pivot_imp

# =============================================================================
# TIEMPO DE CÁLCULO DE LOS MÉTODOS DE IMPUTACIÓN
# =============================================================================

df_time['Recon_aux'] = df_time['Model'].apply(lambda x: x.split("-"))

df_time['Recon']=df_time.apply(lambda x: recon_rows(x), axis=1)
df_time.sort_values('Recon', inplace=True)
nombres = ['Unnamed: 0', 'Model', 'Tiempo de imputación (s)', 'Tiempo de corrección (s)', 
           'Tiempo de predicción (s)', 'Recon_method', 'Total', 'Forecasting_Model', 'Recon_aux', 'Recon']
df_time.columns = nombres

boxplot(category = 'Recon', value='Tiempo de imputación (s)', df=df_time)

# =============================================================================
# SIGNIFICACIÓN ESTADÍSTICA DE LA CORRECCIÓN DE VALORES ANÓMALOS
# =============================================================================

df_res = df_results.copy()
df_res = df_res.replace('Correction_True', 'Serie corregida')
df_res = df_res.replace('Correction_False', 'Serie sin corregir')

stat_test(df_results[df_results['Corrected']=='Correction_True']['RMSE'], 
          df_results[df_results['Corrected']=='Correction_False']['RMSE'], alpha=0.05, crit='levene')

stat_test(df_results[df_results['Corrected']=='Correction_True']['RMSE'], 
          df_results[df_results['Corrected']=='Correction_False']['RMSE'], alpha=0.05, crit='kruskal')

boxplot(category = 'Corrected', value='RMSE', df=df_res)

# =============================================================================
# SIGNIFICACIÓN ESTADÍSTICA DE LA PREDICCIÓN
# =============================================================================

df_p_valor_pred = pd.DataFrame({"Método A": [],
                  'Método B': [],
                  'P-valor':[]})

methods = list(df_results['Forecasting_Model'].unique())
alpha = 0.05
for met_a in methods:
    for met_b in methods:
        sample_a = df_results[df_results['Forecasting_Model']==met_a]['RMSE']
        sample_b = df_results[df_results['Forecasting_Model']==met_b]['RMSE']
        print('**********************************************')
        print(f"PRUEBA DE HIPÓTESIS PARA LOS MÉTODOS {met_a} Y {met_b}")
        df_p_valor_pred = df_p_valor_pred.append({"Método A": met_a,
                  'Método B': met_b,
                  'P-valor': stat_test(sample_a, sample_b, crit='kruskal', alpha=alpha)}, ignore_index=True)
        

pivot_pred = df_p_valor_pred.pivot(index='Método A',
                 columns = 'Método B',
                 values= 'P-valor')

pivot_pred.head()

# =============================================================================
# GRAFICADO DE LOS MEJORES RESULTADOS POR MODELO
# =============================================================================

y_ann = np.loadtxt('./Predictions_v2/median-Correction_False-ANN-96-96.csv')
y_gbr = np.loadtxt('./Predictions_v2/hybrid-HW-mean-Correction_False-GradientBoostingRegressor-96-96.csv')
y_svr = np.loadtxt('./Predictions_v2/hybrid-RF-mean-Correction_False-SVR-96-96.csv')
y_rf = np.loadtxt('./Predictions_v2/median-Correction_True-RandomForestRegressor-96-96.csv')
y_knn = np.loadtxt('./Predictions_v2/hybrid-ARIMA-mean-Correction_False-KNeighborsRegressor-96-96.csv')
y_dt = np.loadtxt('./Predictions_v2/hybrid-KNN-mean-Correction_False-DecisionTreeRegressor-96-96.csv')

x = list(range(96))
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig, ax = plt.subplots(figsize=(20,10))

# Using set_dashes() to modify dashing of an existing line
line1, = ax.plot(x, y_real, label='Caudal real', marker='.')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
line2, = ax.plot(x, y_ann, label='ANN', marker = 'o', markersize=3, color='black')
line3, = ax.plot(x, y_gbr, label='GBR', marker='o', markersize=3)
line4, = ax.plot(x, y_svr, label='SVR', marker='o', markersize=3)
line5, = ax.plot(x, y_rf, label='RF', marker='o', markersize=3)
line6, = ax.plot(x, y_knn, label='KNN', marker='o', markersize=3)
line7, = ax.plot(x, y_dt, label='DT', marker='o', markersize=3)

ax.set_ylabel('Caudal medido (m3/h)', fontsize=18)
ax.set_xlabel('Instante de tiempo en intervalos de 15 minutos', fontsize=18)

ax.legend(fontsize=14)
plt.show()

# =============================================================================
# GRAFICADO DE VALORES ANÓMALOS
# =============================================================================

df_flow = pd.read_csv("Data/flow.txt", parse_dates=[3], sep=';', header = None, skiprows=0).drop([0, 1, 2], axis=1)
pipe = Pipeline(df_flow)
pipe.wrangle(plot=0)
pipe.recon(method='median', weeks=4, resample=False, how='H', short=None, long=None, steps=96, seasonal1=96, seasonal2=96*7)
pipe.outliers(0, 0.05, correction=True)
pipe.day_plot(dia=12, mes=6, ano=2018, alpha=0.05)

# =============================================================================
# PREDICCIÓN CON EL MEJOR MÉTODO
# =============================================================================

df_flow = pd.read_csv("Data/flow.txt", parse_dates=[3], sep=';', header = None, skiprows=0).drop([0, 1, 2], axis=1)
pipe = Pipeline(df_flow)
pipe.wrangle(plot=0)
pipe.recon(method='median', weeks=6, resample=False, how='H', short=None, long=None, steps=96, seasonal1=96, seasonal2=96*7)
pipe.outliers(0, 0.05, correction=False)


iterations = 20
mejor_resultado = np.empty((96, iterations))

for it in range(iterations):
    pipe.predict('ANN', stat=96, horizon=96)
    mejor_resultado[:, it] = pipe._y_forecast
    
mejor_resultado_media = np.mean(mejor_resultado, axis=1)
mejor_resultado_std = np.std(mejor_resultado, axis=1)

res = pd.DataFrame({"Media": mejor_resultado_media,
                   'Inferior': mejor_resultado_media-1.96*mejor_resultado_std,
                   'Superior': mejor_resultado_media+1.96*mejor_resultado_std})

'''
Salida gráfica del mejor resultado con los intervalos de confianza al 95%
'''
fig, ax = plt.subplots(figsize=(20,10))

line1, = ax.plot(x, y_real, label='Caudal real', marker='.')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
line2, = ax.plot(x, res.Media, label='Caudal predicho', marker = 'o', markersize=3, color='black')
line3, = ax.plot(x, res.Inferior, label='Intervalo de confianza al 95%', marker='o', markersize=3, color='red')
line4, = ax.plot(x, res.Superior, marker='o', markersize=3, color='red')

ax.set_ylabel('Caudal medido (m3/h)', fontsize=18)
ax.set_xlabel('Instante de tiempo en intervalos de 15 minutos', fontsize=18)

ax.legend(fontsize=18)
plt.show()