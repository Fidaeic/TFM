# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:27:08 2020

@author: Fidae El Morer

Genetic Algorithm for hyperparameter optimization
"""

import pandas as pd
import numpy as np
import wrangler
import matplotlib.pyplot as plt
from ypstruct import structure

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
from sklearn.datasets import fetch_california_housing

from sklearn.metrics import r2_score
import time
#import xgboost as xgb
from scipy.stats import norm

#Run GA
def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

def crossover(p1, p2, gamma=0.1):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *c1.position.shape)
    c1.position = alpha*p1.position + (1-alpha)*p2.position
    c2.position = alpha*p2.position + (1-alpha)*p1.position
    return c1, c2

def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma*np.random.randn(*ind.shape)
    return y
    
def apply_bound(x, varmin, varmax):
    x.position = np.maximum(x.position, varmin)
    x.position = np.minimum(x.position, varmax)
    
def run(problem, params, X, y, model):

    # Problem Information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc = params.pc
    nc = int(np.round(pc*npop/2)*2)
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = 0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # Initialize Population
    pop = empty_individual.repeat(npop)
    
    if model=='RF':
        for i in range(npop):
            pop[i].position = np.random.randint(varmin, varmax, nvar)
    
            rf = RandomForestRegressor(n_estimators=pop[i].position[0],
                                      max_depth=pop[i].position[1],
                                      min_samples_split=pop[i].position[2],
                                      min_samples_leaf=pop[i].position[3])
            
            rf.fit(X_train, y_train)
            y_hat = rf.predict(X_test)
            
            pop[i].cost = costfunc(y_hat, y_test)
            
            if pop[i].cost > bestsol.cost:
                bestsol = pop[i].deepcopy()
    
        # Best Cost of Iterations
        bestcost = np.empty(maxit)
        
        # Main Loop
        for it in range(maxit):
    
            costs = np.array([x.cost for x in pop])
            avg_cost = np.mean(costs)
            if avg_cost != 0:
                costs = costs/avg_cost
            probs = np.exp(-beta*costs)
    
            popc = []
            for _ in range(nc//2):
    
                # Select Parents
                #q = np.random.permutation(npop)
                #p1 = pop[q[0]]
                #p2 = pop[q[1]]
    
                # Perform Roulette Wheel Selection
                p1 = pop[roulette_wheel_selection(probs)]
                p2 = pop[roulette_wheel_selection(probs)]
                
                # Perform Crossover
                c1, c2 = crossover(p1, p2, gamma)
    
                # Perform Mutation
                c1 = mutate(c1, mu, sigma)
                c2 = mutate(c2, mu, sigma)
    
                # Apply Bounds
                apply_bound(c1, varmin, varmax)
                apply_bound(c2, varmin, varmax)
    
                # Evaluate First Offspring
                rf_1 = RandomForestRegressor(n_estimators=int(c1.position[0]),
                                      max_depth=int(c1.position[1]),
                                      min_samples_split=int(c1.position[2]),
                                      min_samples_leaf=int(c1.position[3]))
            
                rf_1.fit(X_train, y_train)
                y_hat_1 = rf_1.predict(X_test)
                
                c1.cost = costfunc(y_hat_1, y_test)
                if c1.cost > bestsol.cost:
                    bestsol = c1.deepcopy()
    
                # Evaluate Second Offspring
                rf_2 = RandomForestRegressor(n_estimators=int(c2.position[0]),
                                      max_depth=int(c2.position[1]),
                                      min_samples_split=int(c2.position[2]),
                                      min_samples_leaf=int(c2.position[3]))
            
                rf_2.fit(X_train, y_train)
                y_hat_2 = rf_2.predict(X_test)
    
                c2.cost = costfunc(y_hat_2, y_test)
                if c2.cost > bestsol.cost:
                    bestsol = c2.deepcopy()
    
                # Add Offsprings to popc
                popc.append(c1)
                popc.append(c2)
        

        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.cost, reverse=True)
        pop = pop[0:npop]

        # Store Best Cost
        bestcost[it] = bestsol.cost

        # Show Iteration Information
        print("Iteration {}: Best Cost = {}".format(it, bestcost[it]))

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out

def prec(y_hat, y_real):
    return r2_score(y_hat, y_real)

# =============================================================================
# EXECUTION OF CODE
# =============================================================================

# Problem Definition
problem = structure()
problem.costfunc = prec
problem.nvar = 4
#n_estimators, max_depth, min samples split, min samples leaf
problem.varmin = [1, 1, 2, 1]
problem.varmax = [500,  100, 20, 20]

# GA Parameters
params = structure()
params.maxit = 50
params.npop = 50
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1

data = fetch_california_housing()
X, y = data['data'][:200], data['target'][:200]

# Run GA
out = run(problem, params, X, y, model='RF')