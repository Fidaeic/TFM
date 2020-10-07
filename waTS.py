# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:04:33 2020

@author: Propietario
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pred
from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tbats import TBATS
from pmdarima.arima import auto_arima, ADFTest

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

class waTS(object):
    
    def __init__(self, ts):
        
        self.ts = ts
        self.ts_recon = None
        self.ts_matrix = None
        self.median_matrix = None
        self.df_matrix = None
        self.df_outliers = None
        self.df_correct = None
        self.ts_final = None
    
        self._time = 0
        self._nul = None
        self._sto = None
        self._chunks = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None
        self._X = None
        self._y = None
        self._y_for_test = None
        self._accuracy = None
        self._alpha = None
        self._length = None
        self._days  = None

# =============================================================================
# DATA WRANGLING METHODS
# =============================================================================
    def data_wrangler(self, plot):
        
        ts = self.ts
        
        start_time = time.time()
        
        ts.columns = ["DateTime", "Flow"]
        ts = ts.drop_duplicates(subset="DateTime", keep='first')
        ts = ts.sort_values(by="DateTime").reset_index()
        r = pd.date_range(start=ts.DateTime.min(), end=ts.DateTime.max(), freq='15min')
        ts = ts.set_index('DateTime')
        ts = ts.reindex(r)
        ts.drop(labels="index", axis=1, inplace=True)
        null = np.array(ts[ts["Flow"].isna()].index)
        #almacacenamos los datos nulos
        stored = []
        for i in range(1, len(null)):
            delta = null[i]- null[i-1]
            delta = delta.astype('timedelta64[m]')
            if delta != np.timedelta64(15, 'm'):
                stored.append((null[i-1], null[i]))
    
        #Conseguimos los bloques de datos null
        chunks = []
        chunks.append(ts.loc[null[0]:stored[0][0]].index)
        for i in range(0, len(stored)-1):
            chunks.append(ts.loc[stored[i][1]:stored[i+1][0]].index)
        chunks.append(ts.loc[stored[-1][1]:null[-1]].index)

        date_first = str(ts.head(1).index.year[0])+"-"+str(ts.head(1).index.month[0])+"-"+str(ts.head(1).index.day[0])
        date_last = str(ts.tail(1).index.year[0])+"-"+str(ts.tail(1).index.month[0])+"-"+str(ts.tail(1).index.day[0])

        if len(ts.loc[date_first])<96:
            ts.drop(ts.drop.loc[date_first].index, inplace=True)

        if len(ts.loc[date_last])<96:
            ts.drop(ts.loc[date_last].index, inplace=True)

        steps = ts.groupby(ts.index.floor('d')).size()
        length = steps[0]
        days = len(steps)
        
        #Dibujamos en un gráfico lfirstas posiciones en las que hay datos vacíos
        if plot==True:    
            plt.figure(figsize=(30, 10))
            plt.xlabel("Time", fontsize=18)
            plt.xticks(fontsize=14, rotation=45)
            plt.yticks(fontsize=14)
            plt.plot(ts.isna())
        
        elapsed_time = time.time()-start_time
        
        self.ts = ts
        self._nul = null
        self._sto = stored
        self._chunks = chunks
        self._length = length
        self._days = days
        self._time += elapsed_time
        
    def shifting(self, df, stat, horizon, t_size=0.3):
        
        start_time = time.time()
        dataframe = pd.DataFrame()
        for i in range(stat, 0, -1):
            dataframe['t-'+str(i)] = df.Flow.shift(i, fill_value=np.nan)
        
        dataframe['t'] = df.Flow.values
        ind_0 = dataframe[dataframe[f't-{stat}'].isna() == False].index[0]
        aux = dataframe.loc[ind_0:]
        X = aux.iloc[:-horizon, :-1]
        y = aux.iloc[:-horizon, -1]
        y_for_test = aux.iloc[-horizon:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, shuffle=False)
        
        elapsed_time = time.time()-start_time
        
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._X = X
        self._y = y
        self._y_for_test = y_for_test
        self._time += elapsed_time

    def matrix(self, fest, ts):

        start_time = time.time()
        
        length = self._length
        days = self._days
        
        data = np.zeros((days,length+6))
        
        for d, t in zip(range(days), range(0, ts.size, length)):
            data[d,:length] = ts.iloc[t:t+length].values.reshape(length)
            data[d,length+1] = int(ts.index[t].dayofweek+1)
            data[d,length+2] = int(ts.index[t].day)
            data[d,length+3] = int(ts.index[t].month)
            data[d,length+4] = int(ts.index[t].year)
            
            if fest:
                for f in fest:
                    if ts.index[t]==pd.to_datetime(f):
                        data[d,length+5] = 7
        
        data = data[~np.isnan(data).any(axis=1)]
        elapsed_time = time.time()-start_time

        self.ts_matrix = data
        self._time += elapsed_time
# =============================================================================
# CLUSTERING
# =============================================================================
    def Qn(self, x):
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

    def clusters(self):
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
        start_time = time.time()
        mat = self.ts_matrix
        length = self._length
        
        df = pd.DataFrame(mat)
        df['day'] = np.where(np.logical_and(df[length+1]>=1, df[length+1]<=5), 0, 1)

        df['labels'] = np.where(df['day']==0,
                                      np.where(
                                          np.logical_or(df[length+3]==12, df[length+3]<3), 0, np.where(
                                              np.logical_and(df[length+3]>=3, df[length+3]<6), 2, np.where(
                                                  np.logical_and(df[length+3]>=6, df[length+3]<9), 4, 6))), 
                                      np.where(
                                          np.logical_or(df[length+3]==12, df[length+3]<3), 1, np.where(
                                              np.logical_and(df[length+3]>=3, df[length+3]<6), 3, np.where(
                                                  np.logical_and(df[length+3]>=6, df[length+3]<9), 5, 7))))
    
        median = np.zeros((len(np.unique(df.labels)), length))
        for i in np.unique(df.labels):
            a = df[df['labels']==i]
            for j in range(length):
                median[i, j] = np.median(a.iloc[:, j])
        elapsed_time = time.time()-start_time
        self.median_matrix = median
        self.df_matrix = df
        self._time+=elapsed_time
    
    def outlier_region(self, alpha):
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
        start_time = time.time()
        med = self.median_matrix
        cl = self.df_matrix
        df = self.ts_recon
        length = self._length
        
        qn = []
    
        for i in range(med.shape[0]):
            a = np.array(cl[cl['labels']==i].drop(cl.columns[-8:], axis=1))
            b = np.reshape(a,a.size)
            qn.append(self.Qn(b))
    
        upper = []
        lower = []
    
        for row in cl.iterrows():
            a = row[:length]
            cluster = int(row[1]['labels'])
            m = med[cluster, :]
            upper.append([x + norm.ppf(1-alpha/2)*qn[cluster] for x in m])
            lower.append([x - norm.ppf(1-alpha/2)*qn[cluster] for x in m])
    
        upper = np.array(upper)
        upper = upper.reshape(upper.size)
    
        lower = np.array(lower)
        lower = lower.reshape(lower.size)
    
        medianas = []
        for row in cl.iterrows():
            medianas.append(med[int(row[1]['labels'])])
            
        medianas = np.array(medianas)
        medianas = medianas.reshape(medianas.size)
        
        dataframe = pd.DataFrame()
        
        a = (1-alpha)*100
        
        dataframe["Flow"] = df.Flow
        dataframe[f"Upper_{a}"] = upper
        dataframe[f"Lower_{a}"] = lower
    
        dataframe["Median"] = medianas

        elapsed_time = time.time()-start_time
        self.df_outliers = dataframe
        self._time += elapsed_time
        self._alpha = alpha
        
    def corrected(self):
        
        df_outliers = self.df_outliers
        start_time = time.time()
        alpha = self._alpha
        
        a = (1-alpha)*100
        df_outliers.loc[(df_outliers['Flow']>df_outliers[f'Upper_{a}']), 'Flow'] = df_outliers.loc[(df_outliers['Flow']>df_outliers[f'Upper_{a}']), 'Median']
        df_outliers.loc[(df_outliers['Flow']<df_outliers[f'Lower_{a}']), 'Flow'] = df_outliers.loc[(df_outliers['Flow']<df_outliers[f'Lower_{a}']), 'Median']
        
        df_correct=df_outliers.copy()
        '''
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
        '''
        elapsed_time = time.time()-start_time
        
        self.df_correct = df_correct
        self._time += elapsed_time
    
    def day_plot(self, dia, mes, ano, alpha):
    
        df = self.df_outliers
    
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
# =============================================================================
# FORECASTING METHODS
# =============================================================================
    def forecast(self, ts, model, stat, horizon, prt):
        '''
        Parameters
        ----------
        model :  Sklearn objects
            Machine Learning methods present in the Sklearn library.
        stat : int, optional
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
        
        self.shifting(ts, stat, horizon)
    
        X_train = self._X_train
        X_test = self._X_test
        y_train = self._y_train
        y_test = self._y_test
        X = self._X
        y = self._y 
        y_for_test = self._y_for_test
        
        model.fit(X=X_train, y=y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        X_for = X.iloc[-1, 1:]
        X_for['t'] = y.iloc[-1]
        X_for = np.array(X_for).reshape(1, stat)
        df_for = pd.DataFrame(X_for)
        df_for[stat] = model.predict(np.array(df_for.iloc[0]).reshape(1, stat))
    
        for i in range(1, horizon):
            df_for = df_for.append(df_for.iloc[i-1, :].shift(-1))
            df_for.iloc[i, stat] = model.predict(np.array(df_for.iloc[i, :-1]).reshape(1, stat))
    
        y_forecast = df_for[stat].values
        
        acc = {"Training R2": r2_score(y_train, y_pred_train)*100,
                "Test R2": r2_score(y_test, y_pred_test)*100,
                "Forecast R2": r2_score(y_for_test, y_forecast)*100}

        # Training confidence intervals
        # res_train = y_pred_train-y_train
        # mu_train = np.mean(res_train)
        # sigma_train = np.std(res_train)
        
        elapsed_time = time.time()-start_time

        if prt==1:
            plt.figure(figsize=(20, 10))
            xlabels = y_for_test.index
            plt.plot(y_for_test.values, color='black')
            plt.plot(y_forecast, color='blue')
            #plt.xticks(labels=xlabels, rotation=40, ha='right')
            plt.ylabel("Consumo de agua medido en m3", fontsize=16)
            plt.legend(["Consumo real", "Consumo predicho"], fontsize=14)
        
        self._y_pred_train = y_pred_train
        self._y_pred_test = y_pred_test
        self._y_forecast = y_forecast
        self._accuracy = acc
        self._time += elapsed_time
        
    def forecast_NN(self, ts, nodes, epochs, stat, horizon, lay, init, act, opt, prt):
        '''
        Parameters
        ----------
        ts : Pandas Dataframe
            Dataframe with only one column called "Flow" and and a DateTime index
        nodes : int
            Number of nodes in each layer.
        epochs : int
            Number of epochs.
        stat : int, optional
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
            model.add(Dense(nodes, input_dim=stat, kernel_initializer=init, activation=act))
            
        model.add(Dense(1, kernel_initializer=init))
        # Compile model
        model.compile(optimizer=opt, loss='mse')    
    
        self.shifting(ts, stat, horizon)
    
        X_train = self._X_train
        X_test = self._X_test
        y_train = self._y_train
        y_test = self._y_test
        X = self._X
        y = self._y 
        y_for_test = self._y_for_test
    
        # estimator = KerasRegressor(build_fn=NN_model, nb_epoch=100, batch_size=32, verbose=False)
    
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
        X_for = X.iloc[-1, 1:]
        X_for['t'] = y.iloc[-1]
        X_for = np.array(X_for).reshape(1, stat)
        df_for = pd.DataFrame(X_for)
        df_for[stat] = model.predict(np.array(df_for.iloc[0]).reshape(1, stat))
    
        for i in range(1, horizon):
            df_for = df_for.append(df_for.iloc[i-1, :].shift(-1))
            df_for.iloc[i, stat] = model.predict(np.array(df_for.iloc[i, :-1]).reshape(1, stat))[0]
    
        y_forecast = df_for[stat].values
        
        acc = {"Training R2": r2_score(y_train, y_pred_train)*100,
                "Test R2": r2_score(y_test, y_pred_test)*100,
                "Forecast R2": r2_score(y_for_test, y_forecast)*100}
        
        elapsed_time = time.time()-start_time
        
        if prt==1:
            plt.figure(figsize=(20, 10))
            xlabels = y_for_test.index
            plt.plot(y_for_test.values, color='black')
            plt.plot(y_forecast, color='blue')
            #plt.xticks(labels=xlabels, rotation=40, ha='right')
            plt.ylabel("Consumo de agua medido en m3", fontsize=16)
            plt.legend(["Consumo real", "Consumo predicho"], fontsize=14)
        
        self._y_pred_train = y_pred_train
        self._y_pred_test = y_pred_test
        self._y_forecast = y_forecast
        self._accuracy = acc
        self._time += elapsed_time

# =============================================================================
# RECONSTRUCTION METHODS FOR TIME SERIES.
# The first two methods use the median or the mean of the last n weeks to impute missing values.
# The recon_hybrid method uses one of the first methods when the lenght of the chunk of missing data
# is larger than the forecasting horizon. For instance, if the forecasting horizon is 24 hours and the chunk
# of missing data is larger, the imputation method will be based on the mean or the median, whereas if the chunk
# of missing data is smaller, the imputation method will use time series forecasting methods that can be based on
# machine learning methods or classic time series algorithms.
# =============================================================================

    def recon_median(self, weeks):
        '''
        Parameters
        ----------
        weeks : int, optional
            Number of weeks to consider when imputing missing values. The default is 6.
    
        Returns
        -------
        dataframe : Pandas dataframe
            Dataframe with imputed values considering the median values of the last n weeks
        elapsed_time : float
            Elapsed time to perform the imputing method.
        '''
        
        ts = self.ts
        nul = self._nul
        
        start_time = time.time()
        dataframe = ts.copy()
        
        for n in nul:
            values = []
            for k in range(weeks):
                values.append(dataframe.loc[n-pd.Timedelta(value=(k+1)*7, unit='D')])
            dataframe.loc[n] = np.median(values)
        elapsed_time = time.time()-start_time
        
        self.ts_recon = dataframe
        self._time += elapsed_time
        
    def recon_mean(self, weeks):
        '''
        Parameters
        ----------
        weeks : int, optional
            Number of weeks to consider when imputing missing values.
        Returns
        -------
        dataframe : Pandas dataframe
            Dataframe with imputed values considering the mean values of the last n weeks
        elapsed_time : float
            Elapsed time to perform the imputing method.
        '''
        ts = self.ts
        nul = self._nul
        
        start_time = time.time()
        dataframe = ts.copy()
        
        for n in nul:
            values = []
            for k in range(weeks):
                values.append(dataframe.loc[n-pd.Timedelta(value=(k+1)*7, unit='D')])
            dataframe.loc[n] = np.mean(values)
        elapsed_time = time.time()-start_time
        
        self.ts_recon = dataframe
        self._time += elapsed_time
        
    def recon_hybrid(self, steps, seasonal1, seasonal2, short, long, weeks):
        '''
        Parameters
        ----------
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
        ts = self.ts
        chunks = self._chunks

        start_time = time.time()
        dataframe = ts.copy()

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

                elif short == 'HW':
                    estimator = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=seasonal1)
                    fitted_model = estimator.fit()
                    y_forecast = fitted_model.forecast(steps=len(c))

                elif short == 'KNN':
                    self.forecast(ts, KNeighborsRegressor(), horizon=len(c), stat=seasonal1, prt=0)
                    y_forecast = self._y_forecast
                elif short == 'RF':
                    self.forecast(ts, RandomForestRegressor(), horizon=len(c), stat=seasonal1, prt=0)
                    y_forecast = self._y_forecast
                elif short == 'SVR':
                    self.forecast(ts, SVR(), horizon=len(c), stat=seasonal1, prt=0)
                    y_forecast = self._y_forecast
    
                j=0
                for n in c:
                    dataframe.loc[n] = y_forecast[j]
                    j+=1
        elapsed_time = time.time()-start_time
        
        self.ts_recon = dataframe
        self._time += elapsed_time

    def resample(self, how):
        ts_recon_resample = self.ts_recon.resample(how).median()
        steps = ts_recon_resample.groupby(ts_recon_resample.index.floor('d')).size()
        length = steps[0]
        days = len(steps)
        
        self.ts_recon = ts_recon_resample
        self._length = length
        self._days = days
        
class Pipeline(waTS):

    def wrangle(self, plot):
        self.data_wrangler(plot=plot)

    def recon(self, method, weeks, resample, how, short, long, steps, seasonal1, seasonal2):
        if method=='mean':
            self.recon_mean(weeks)
        elif method=='median':
            self.recon_median(weeks)
        else:
            self.recon_hybrid(steps, seasonal1, seasonal2, short, long, weeks)
            
        if resample==True:
            self.resample(how=how)

    def outliers(self, fest, alpha, correction=True):
        self.matrix(fest, self.ts_recon)
        self.clusters()
        self.outlier_region(alpha)
        if correction==True:
            self.corrected()
            self.ts_final = self.df_correct
        else:
            self.ts_final = self.ts_recon

    def predict(self, model, stat, horizon, prt=0, 
                 nodes=20, epochs=50, lay=3, init='normal', act='relu', opt='adam', nn=False):
        if model=='NN':
            self.forecast_NN(self.ts_final, nodes, epochs, stat, horizon, lay, init, act, opt, prt)
        else:
            self.forecast(self.ts_final, model, stat, horizon, prt)
            