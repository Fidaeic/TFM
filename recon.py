# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:57:52 2020

@author: Fidae El Morer
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tbats import TBATS
from pmdarima.arima import auto_arima, ADFTest
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import pred

# =============================================================================
# RECONSTRUCTION METHODS FOR TIME SERIES.
# The first two methods use the median or the mean of the last n weeks to impute missing values.
# The recon_hybrid method uses one of the first methods when the lenght of the chunk of missing data
# is larger than the forecasting horizon. For instance, if the forecasting horizon is 24 hours and the chunk
# of missing data is larger, the imputation method will be based on the mean or the median, whereas if the chunk
# of missing data is smaller, the imputation method will use time series forecasting methods that can be based on
# machine learning methods or classic time series algorithms.
# =============================================================================
def recon_median(df, nul, weeks=6):
    '''
    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe with only one column called "Flow" and and a DateTime index
    nul : list
        List with the missing values. It does not take into account separated chunks.
    weeks : int, optional
        Number of weeks to consider when imputing missing values. The default is 6.

    Returns
    -------
    dataframe : Pandas dataframe
        Dataframe with imputed values considering the median values of the last n weeks
    elapsed_time : float
        Elapsed time to perform the imputing method.
    '''
    start_time = time.time()
    dataframe = df.copy()
    for n in nul:
        values = []
        for k in range(weeks):
            values.append(dataframe.loc[n-pd.Timedelta(value=(k+1)*7, unit='D')])
        dataframe.loc[n] = np.median(values)
    elapsed_time = time.time()-start_time
    return dataframe, elapsed_time

def recon_mean(df, nul, weeks=6):
    '''
    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe with only one column called "Flow" and and a DateTime index
    nul : list
        List with the chunks of missing data.
    weeks : int, optional
        Number of weeks to consider when imputing missing values. The default is 6.

    Returns
    -------
    dataframe : Pandas dataframe
        Dataframe with imputed values considering the mean values of the last n weeks
    elapsed_time : float
        Elapsed time to perform the imputing method.
    '''
    start_time = time.time()
    dataframe = df.copy()
    for n in nul:
        values = []
        for k in range(weeks):
            values.append(dataframe.loc[n-pd.Timedelta(value=(k+1)*7, unit='D')])
        dataframe.loc[n] = np.mean(values)
    elapsed_time = time.time()-start_time
    return dataframe, elapsed_time

def recon_hybrid(df, chunks, steps, seasonal1=96, seasonal2=672, short='ARIMA', long='median', weeks=6):
    '''
    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe with only one column called "Flow" and and a DateTime index
    chunks : list
        List with the chunks of missing values, which are lists as well. This variable is returned by 
        the function wrangler.data_wrangler.
    steps : int
        Maximum number of steps that are going to be forecasted.
    seasonal1 : int, optional
        First seasonality of the time series. The default is 96, considering flow values every 15 minutes during a day.
    seasonal2 : int, optional
        Second seasonality of the time series. It is not used by all the methods. 
        The default is 672, considering flow values every 15 minutes during a week.
    short : string, optional
        Defines the method used to impute whenever the chunk of missing data is smaller than the forecasting horizon. 
        The default is 'ARIMA'.
    long : string, optional
        Same as "short", but regarding chunks larger than the forecasting horizon. The default is 'median'.
    weeks : int, optional
        Number of weeks to consider when imputing missing values. The default is 6.

    Returns
    -------
    dataframe : Pandas Dataframe
        Dataframe with imputed values according to the selected methods.
    elapsed_time : float
        Elapsed time to perform the imputing method.
    '''
    start_time = time.time()
    dataframe = df.copy()
    for c in chunks:
        if len(c)>steps:
            for n in c:
                values = []
                for k in range(weeks):
                    values.append(dataframe.loc[n-pd.Timedelta(value=(k+1)*7, unit='D')]) 

                if long == 'median':
                    dataframe.loc[n] = np.median(values)
                elif long == 'mean':
                    dataframe.loc[n] = np.mean(values)

        else:
            ts = dataframe.loc[:c[0]].iloc[:-1]            
            if short == 'ARIMA':
                arima_model = auto_arima(ts)
                y_forecast = arima_model.predict(n_periods=len(c))
                
            elif short == 'TBATS':
                estimator = TBATS(seasonal_periods=[seasonal1, seasonal2])
                fitted_model = estimator.fit(ts)
                y_forecast = fitted_model.forecast(steps=len(c))
            
            elif short == 'HW':
                estimator = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=seasonal1)
                fitted_model = estimator.fit()
                y_forecast = fitted_model.forecast(steps=len(c))
            elif short == 'KNN':
                res = pred.forecast(ts, KNeighborsRegressor(), horizon=len(c), estac=seasonal1, prt=0)
                y_forecast = res[3]
            elif short == 'RF':
                res = pred.forecast(ts, RandomForestRegressor(), horizon=len(c), estac=seasonal1, prt=0)
                y_forecast = res[3]
            elif short == 'SVR':
                res = pred.forecast(ts, SVR(), horizon=len(c), estac=seasonal1, prt=0)
                y_forecast = res[3]
            elif short == 'GPR':
                res = pred.forecast(ts, GaussianProcessRegressor(), horizon=len(c), estac=seasonal1, prt=0)
                y_forecast = res[3]

            j=0
            for n in c:
                dataframe.loc[n] = y_forecast[j]
                j+=1
    elapsed_time = time.time()-start_time
    return dataframe, elapsed_time
