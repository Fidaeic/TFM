# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:30:18 2020

@author: Propietario
"""

from waTS import waTS
import pandas as pd

df_flow = pd.read_csv("Data/flow.txt", parse_dates=[3], sep=';', header = None, skiprows=0).drop([0, 1, 2], axis=1)

prueba = waTS(df_flow)

prueba.data_wrangler(plot=True)
prueba.recon_median(6)

# prueba.ts_recon.plot()

# prueba.forecast(prueba.ts_recon, KNeighborsRegressor(), 96, 96, 1)

# prueba.forecast_NN(prueba.ts_recon, nodes=20, epochs=20, stat=96, horizon=96, lay=5, init='normal', act='relu', opt='adam', prt=1)

prueba.matrix(0, prueba.ts_recon.drop(prueba.ts_recon.tail(4).index))

prueba.clusters()

bzz = prueba.outlier_region()

# def Qn(x):
#     '''
#     Parameters
#     ----------
#     x : numpy array or list
#         DESCRIPTION.

#     Returns
#     -------
#     qn : float
#         Qn estimator.
#     '''

#     n = len(x)
#     h = (n//2)+1
#     k = (h*(h-1))//2

#     ser = []

#     for i in range(len(x)):
#         for j in range(1, len(x)):
#             if j>i:
#                 ser.append(abs(x[i]-x[j]))
#     ser = sorted(ser)
#     qn = 2.2219*ser[k]
#     return qn

# med = prueba.median_matrix
# cl = prueba.df_matrix
# qn = []
# for i in range(med.shape[0]):
#     a = np.array(cl[cl['labels']==i].drop(cl.columns[-8:], axis=1))
#     b = np.reshape(a,a.size)
#     qn.append(Qn(b))