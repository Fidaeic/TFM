# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:27:07 2020

@author: Propietario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_time = pd.read_csv('Time.csv')
df_results = pd.read_csv('Results.csv')
y_real = np.loadtxt('Y_real.csv')


def boxplot(category, value, df):

    sns.set_style(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x=value, y=category, data=df,
                whis=[0, 100], width=.6, palette="vlag")

    # Add in points to show each observation
    sns.stripplot(x=value, y=category, data=df,
                  size=4, color=".3", linewidth=0)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    #sns.despine(trim=True, left=True)

boxplot(category = 'Recon_method', value='RMSE', df=df_results)

boxplot(category = 'Recon_method', value='Forecast', df=df_time)



boxplot(category = 'Corrected', value='RMSE', df=df_results)



boxplot(category = 'Forecasting_Model', value='RMSE', df=df_results)

boxplot(category = 'Forecasting_Model', value='Forecast', df=df_time)

