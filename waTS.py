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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor

class waTS(object):
    
    def __init__(self, ts):
        
        self.ts = ts
        self.ts_recon = None
        self.ts_matrix = None
    
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
    
        #Dibujamos en un gráfico las posiciones en las que hay datos vacíos
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
        days = len(ts.groupby(ts.index.floor('d')).size())
        
        data = np.zeros((days,102))
        
        for d, t in zip(range(days), range(0, ts.size, 96)):
            data[d,:96] = ts.iloc[t:t+96].values.reshape(96)
            data[d,97] = int(ts.index[t].dayofweek+1)
            data[d,98] = int(ts.index[t].day)
            data[d,99] = int(ts.index[t].month)
            data[d,100] = int(ts.index[t].year)
            
            if fest:
                for f in fest:
                    if ts.index[t]==pd.to_datetime(f):
                        data[d,101] = 7
        
        data = data[~np.isnan(data).any(axis=1)]
        elapsed_time = time.time()-start_time

        self.ts_matrix = data
        self._time += elapsed_time

# =============================================================================
# CLUSTERING
# =============================================================================
    def Qn(x):
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

    def clusters(mat, n_clusters):
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

        lab_win = mat[np.where(np.logical_and(np.logical_and(mat[:,97]>=1, mat[:,97]<=5), np.logical_and(mat[:,99]>=12, mat[:,99]<3)))]
        fest_win = mat[np.where(np.logical_and(np.logical_or(mat[:,97]==6, mat[:,97]==7), np.logical_and(mat[:,99]>=12, mat[:,99]<3)))]

        lab_spr = mat[np.where(np.logical_and(np.logical_and(mat[:,97]>=1, mat[:,97]<=5), np.logical_and(mat[:,99]>=3, mat[:,99]<6)))]
        fest_spr = mat[np.where(np.logical_and(np.logical_or(mat[:,97]==6, mat[:,97]==7), np.logical_and(mat[:,99]>=3, mat[:,99]<6)))]

        lab_sum = mat[np.where(np.logical_and(np.logical_and(mat[:,97]>=1, mat[:,97]<=5), np.logical_and(mat[:,99]>=6, mat[:,99]<9)))]
        fest_sum = mat[np.where(np.logical_and(np.logical_or(mat[:,97]==6, mat[:,97]==7), np.logical_and(mat[:,99]>=6, mat[:,99]<9)))]

        lab_fal = mat[np.where(np.logical_and(np.logical_and(mat[:,97]>=1, mat[:,97]<=5), np.logical_and(mat[:,99]>=9, mat[:,99]<12)))]
        fest_fal = mat[np.where(np.logical_and(np.logical_or(mat[:,97]==6, mat[:,97]==7), np.logical_and(mat[:,99]>=9, mat[:,99]<12)))]
        #Hay que ponerle una etiqueta a los datos para que sean más interpretables y podamos aplicarles las líneas de abajo
        
        
        clus = KMeans(n_clusters=n_clusters).fit(mat)
        df = pd.DataFrame(mat)
        df['labels'] = clus.labels_
        mat = np.array(df)
    
        median = np.zeros((len(np.unique(clus.labels_)), 96))
        for i in np.unique(clus.labels_):
            a = mat[mat[:,-1]==i]
            for j in range(96):
                median[i, j] = np.median(a[:, j])
        return mat, median, clus.inertia_
    
    def outlier_region(df, n_clusters):
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
        mat = self.matrix(0, df)
        cl, med, inertia = self.clusters(mat, n_clusters)
        qn = []
    
        for i in range(med.shape[0]):
            a = cl[cl[:,-1]==i][:,:96]
            b = np.reshape(a,a.size)
            qn.append(Qn(b))
    
        upper_90 = []
        lower_90 = []
    
        upper_95 = []
        lower_95 = []
    
        upper_99 = []
        lower_99 = []
    
        for row in cl:
            a = row[:96]
            cluster = int(row[102])
            m = med[cluster, :]
            upper_90.append([x + norm.ppf(1-0.1/2)*qn[cluster] for x in m])
            lower_90.append([x - norm.ppf(1-0.1/2)*qn[cluster] for x in m])
    
            upper_95.append([x + norm.ppf(1-0.05/2)*qn[cluster] for x in m])
            lower_95.append([x - norm.ppf(1-0.05/2)*qn[cluster] for x in m])
    
            upper_99.append([x + norm.ppf(1-0.01/2)*qn[cluster] for x in m])
            lower_99.append([x - norm.ppf(1-0.01/2)*qn[cluster] for x in m])
    
        upper_90 = np.array(upper_90)
        upper_90 = upper_90.reshape(upper_90.size)
    
        lower_90 = np.array(lower_90)
        lower_90 = lower_90.reshape(lower_90.size)
        
        upper_95 = np.array(upper_95)
        upper_95 = upper_95.reshape(upper_95.size)
    
        lower_95 = np.array(lower_95)
        lower_95 = lower_95.reshape(lower_95.size)
        
        upper_99 = np.array(upper_99)
        upper_99 = upper_99.reshape(upper_99.size)
    
        lower_99 = np.array(lower_99)
        lower_99 = lower_99.reshape(lower_99.size)
    
        medianas = []
        for row in cl:
            medianas.append(med[int(row[102])])
            
        medianas = np.array(medianas)
        medianas = medianas.reshape(medianas.size)
        
        dataframe = pd.DataFrame()
        
        dataframe["Flow"] = df.Flow
        dataframe["Upper_90"] = upper_90
        dataframe["Lower_90"] = lower_90
    
        dataframe["Upper_95"] = upper_95
        dataframe["Lower_95"] = lower_95
    
        dataframe["Upper_99"] = upper_99
        dataframe["Lower_99"] = lower_99
    
        dataframe["Median"] = medianas
        return dataframe
    
    def graficado(dia, mes, ano, df, alpha):
    
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
                    
                elif short == 'TBATS':
                    estimator = TBATS(seasonal_periods=[seasonal1, seasonal2])
                    fitted_model = estimator.fit(ts)
                    y_forecast = fitted_model.forecast(steps=len(c))
                
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
                elif short == 'GPR':
                    self.forecast(ts, GaussianProcessRegressor(), horizon=len(c), stat=seasonal1, prt=0)
                    y_forecast = self._y_forecast
    
                j=0
                for n in c:
                    dataframe.loc[n] = y_forecast[j]
                    j+=1
        elapsed_time = time.time()-start_time
        
        self.ts_recon = dataframe
        self._time += elapsed_time