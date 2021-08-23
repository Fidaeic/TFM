# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:45:49 2020

@author: Propietario
"""

import waTS
import ga
import pandas as pd
import numpy as np
from ypstruct import structure

weeks = 6
resample=False
hor = 96
steps = 96
stat = 96

df_flow = pd.read_csv("Data/flow.txt", parse_dates=[3], sep=';', header = None, skiprows=0).drop([0, 1, 2], axis=1)
pipe = waTS.Pipeline(df_flow)
pipe.wrangle(plot=0)
pipe.recon('median', weeks, resample=resample, how='H', short=None, long=None, steps=hor, seasonal1=stat, seasonal2=stat*7)
dataf = pipe.ts_recon.copy()

problem = structure()
problem.costfunc = ga.prec

problem.nvar = 4
#n_estimators, max_depth, min samples split, min samples leaf
problem.varmin = [1, 1, 2, 1]
problem.varmax = [500,  500, 50, 50]

# GA Parameters
params = structure()
params.maxit = 1
params.npop = 1
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1

# Run GA
optimal = ga.run(problem, params, dataf, 96, 96, model='RF')

print(optimal)