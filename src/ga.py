# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:27:08 2020

@author: Fidae El Morer

Genetic Algorithm for hyperparameter optimization
"""

import numpy as np
import wrangler
from ypstruct import structure
from metrics import rmse

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

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

def prec(y_hat, y_real):
    return rmse(y_hat, y_real)

def knn_weights(num):
    if num==1:
        return 'uniform'
    else:
        return 'distance'

def knn_algorithm(num):
    if num==1:
        return 'auto'
    elif num==2:
        return 'ball_tree'
    elif num==3:
        return 'kd_tree'
    else:
        return 'brute'

def svr_kernel(num):
    if num==1:
        return 'linear'
    elif num==2:
        return 'poly'
    elif num==3:
        return 'rbf'
    elif num==4:
        return 'sigmoid'
    else:
        return 'precomputed'
    
def svr_gamma(num):
    if num==1:
        return 'scale'
    else:
        return 'auto'

def svr_shrinking(num):
    if num==0:
        return True
    else:
        return False
    
def dt_criterion(num):
    if num==1:
        return 'mse'
    elif num==2:
        return 'friedman_mse'
    else:
        return 'mae'

def dt_splitter(num):
    if num==1:
        return 'best'
    else:
        return 'random'
    
def gbr_loss(num):
    if num==1:
        return 'ls'
    elif num==2:
        return 'lad'
    elif num==3:
        return 'huber'
    else:
        return 'quantile'

def gbr_criterion(num):
    if num==1:
        return 'mse'
    elif num==2:
        return 'friedman_mse'
    else:
        return 'mae'

def run(problem, params, df_recon, estac, horizon, model):

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
    bestsol.cost = 1000 

    X_train, X_test, y_train, y_test, X, y, y_for_test= wrangler.shifting(df_recon, estac, horizon)

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
            
            if pop[i].cost<bestsol.cost:
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
                if c1.cost < bestsol.cost:
                    bestsol = c1.deepcopy()
    
                # Evaluate Second Offspring
                rf_2 = RandomForestRegressor(n_estimators=int(c2.position[0]),
                                      max_depth=int(c2.position[1]),
                                      min_samples_split=int(c2.position[2]),
                                      min_samples_leaf=int(c2.position[3]))
            
                rf_2.fit(X_train, y_train)
                y_hat_2 = rf_2.predict(X_test)
    
                c2.cost = costfunc(y_hat_2, y_test)
                if c2.cost < bestsol.cost:
                    bestsol = c2.deepcopy()
    
                # Add Offsprings to popc
                popc.append(c1)
                popc.append(c2)
        
    elif model=='KNN':
        for i in range(npop):
            pop[i].position = np.random.randint(varmin, varmax, nvar)
            
            pop[i].position[4] = np.random.uniform(1, 2, 1)[0]

            knn = KNeighborsRegressor(n_neighbors=pop[i].position[0],
                                      weights=knn_weights(pop[i].position[1]),
                                      algorithm=knn_algorithm(pop[i].position[2]),
                                      leaf_size=pop[i].position[3],
                                      p=pop[i].position[4],
                                      n_jobs=-1)

            knn.fit(X_train, y_train)
            y_hat = knn.predict(X_test)
            
            pop[i].cost = costfunc(y_hat, y_test)
            
            if pop[i].cost<bestsol.cost:
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
                knn_1 = KNeighborsRegressor(n_neighbors=int(c1.position[0]),
                                      weights=knn_weights(int(c1.position[1])),
                                      algorithm=knn_algorithm(int(c1.position[2])),
                                      leaf_size=int(c1.position[3]),
                                      p=c1.position[4],
                                      n_jobs=-1)

            
                knn_1.fit(X_train, y_train)
                y_hat_1 = knn_1.predict(X_test)
                
                c1.cost = costfunc(y_hat_1, y_test)
                if c1.cost < bestsol.cost:
                    bestsol = c1.deepcopy()
    
                # Evaluate Second Offspring
                knn_2 = KNeighborsRegressor(n_neighbors=int(c2.position[0]),
                                      weights=knn_weights(int(c2.position[1])),
                                      algorithm=knn_algorithm(int(c2.position[2])),
                                      leaf_size=int(c2.position[3]),
                                      p=c2.position[4],
                                      n_jobs=-1)
            
                knn_2.fit(X_train, y_train)
                y_hat_2 = knn_2.predict(X_test)
    
                c2.cost = costfunc(y_hat_2, y_test)
                if c2.cost < bestsol.cost:
                    bestsol = c2.deepcopy()
    
                # Add Offsprings to popc
                popc.append(c1)
                popc.append(c2)

    elif model=='SVR':
        for i in range(npop):
            #kernel, degree, gamma, coef0, shrinking
            pop[i].position = np.random.randint(varmin, varmax, nvar)
            
            pop[i].position[4] = np.random.uniform(1, 2, 1)[0]

            svr = SVR(kernel=svr_kernel(pop[i].position[0]),
                        degree=pop[i].position[1],
                        gamma=svr_gamma(pop[i].position[2]),
                        coef0=pop[i].position[3],
                        shrinking=svr_shrinking(pop[i].position[4])
                        )

            svr.fit(X_train, y_train)
            y_hat = svr.predict(X_test)
            
            pop[i].cost = costfunc(y_hat, y_test)
            
            if pop[i].cost<bestsol.cost:
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
                #kernel, degree, gamma, coef0, shrinking
                svr_1 = SVR(kernel=svr_kernel(int(c1.position[0])),
                                      degree=int(c1.position[1]),
                                      gamma=svr_gamma(int(c1.position[2])),
                                      coef0=c1.position[3],
                                      shrinking=svr_shrinking(c1.position[4])
                                      )

            
                svr_1.fit(X_train, y_train)
                y_hat_1 = svr_1.predict(X_test)
                
                c1.cost = costfunc(y_hat_1, y_test)
                if c1.cost < bestsol.cost:
                    bestsol = c1.deepcopy()
    
                # Evaluate Second Offspring
                svr_2 = SVR(kernel=svr_kernel(int(c2.position[0])),
                                      degree=int(c2.position[1]),
                                      gamma=svr_gamma(int(c2.position[2])),
                                      coef0=c2.position[3],
                                      shrinking=svr_shrinking(c2.position[4])
                                      )
            
                svr_2.fit(X_train, y_train)
                y_hat_2 = svr_2.predict(X_test)
    
                c2.cost = costfunc(y_hat_2, y_test)
                if c2.cost < bestsol.cost:
                    bestsol = c2.deepcopy()
    
                # Add Offsprings to popc
                popc.append(c1)
                popc.append(c2)

    elif model=='DT':
        for i in range(npop):
            #criterion, splitter, max_depth, min_samples_split, min_samples_leaf
            pop[i].position = np.random.randint(varmin, varmax, nvar).astype(np.float)
            
            pop[i].position[4] = np.random.uniform(0.0001, 0.5, 1)[0]

            dt = DecisionTreeRegressor(criterion=dt_criterion(int(pop[i].position[0])),
                        splitter=dt_splitter(int(pop[i].position[1])),
                        max_depth=int(pop[i].position[2]),
                        min_samples_split=int(pop[i].position[3]),
                        min_samples_leaf=pop[i].position[4]
                        )

            dt.fit(X_train, y_train)
            y_hat = dt.predict(X_test)
            
            pop[i].cost = costfunc(y_hat, y_test)
            
            if pop[i].cost<bestsol.cost:
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
                #criterion, splitter, max_depth, min_samples_split, min_samples_leaf
                dt_1 = DecisionTreeRegressor(criterion=dt_criterion(int(c1.position[0])),
                                      splitter=dt_splitter(int(c1.position[1])),
                                      max_depth=int(c1.position[2]),
                                      min_samples_split=int(c1.position[3]),
                                      min_samples_leaf=c1.position[4]
                                      )

            
                dt_1.fit(X_train, y_train)
                y_hat_1 = dt_1.predict(X_test)
                
                c1.cost = costfunc(y_hat_1, y_test)
                if c1.cost < bestsol.cost:
                    bestsol = c1.deepcopy()
    
                # Evaluate Second Offspring
                dt_2 = DecisionTreeRegressor(criterion=dt_criterion(int(c2.position[0])),
                                      splitter=dt_splitter(int(c2.position[1])),
                                      max_depth=int(c2.position[2]),
                                      min_samples_split=int(c2.position[3]),
                                      min_samples_leaf=c2.position[4]
                                      )
            
                dt_2.fit(X_train, y_train)
                y_hat_2 = dt_2.predict(X_test)
    
                c2.cost = costfunc(y_hat_2, y_test)
                if c2.cost < bestsol.cost:
                    bestsol = c2.deepcopy()
    
                # Add Offsprings to popc
                popc.append(c1)
                popc.append(c2)

    elif model=='GBR':
        for i in range(npop):
            #loss, learning_rate, n_estimators, criterion, min_samples_split, min_samples_leaf, max_depth
            pop[i].position = np.random.randint(varmin, varmax, nvar).astype(np.float)
            
            pop[i].position[1] = np.random.uniform(0.1, 1, 1)[0]

            gbr = GradientBoostingRegressor(loss=gbr_loss(int(pop[i].position[0])),
                        learning_rate=pop[i].position[1],
                        n_estimators=int(pop[i].position[2]),
                        criterion=gbr_criterion(int(pop[i].position[3])),
                        min_samples_split=int(pop[i].position[4]),
                        min_samples_leaf=int(pop[i].position[5]),
                        max_depth=int(pop[i].position[6])
                        )

            gbr.fit(X_train, y_train)
            y_hat = gbr.predict(X_test)
            
            pop[i].cost = costfunc(y_hat, y_test)
            
            if pop[i].cost<bestsol.cost:
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
                #loss, learning_rate, n_estimators, criterion, min_samples_split, min_samples_leaf, max_depth
                gbr_1 = GradientBoostingRegressor(loss=gbr_loss(int(c1.position[0])),
                        learning_rate=c1.position[1],
                        n_estimators=int(c1.position[2]),
                        criterion=gbr_criterion(int(c1.position[3])),
                        min_samples_split=c1.position[4],
                        min_samples_leaf=c1.position[5],
                        max_depth=int(c1.position[6])
                        )

            
                gbr_1.fit(X_train, y_train)
                y_hat_1 = gbr_1.predict(X_test)
                
                c1.cost = costfunc(y_hat_1, y_test)
                if c1.cost < bestsol.cost:
                    bestsol = c1.deepcopy()
    
                # Evaluate Second Offspring
                gbr_2 = GradientBoostingRegressor(loss=gbr_loss(int(c2.position[0])),
                        learning_rate=c2.position[1],
                        n_estimators=int(c2.position[2]),
                        criterion=gbr_criterion(int(c2.position[3])),
                        min_samples_split=c2.position[4],
                        min_samples_leaf=c2.position[5],
                        max_depth=int(c2.position[6])
                        )
            
                gbr_2.fit(X_train, y_train)
                y_hat_2 = gbr_2.predict(X_test)
    
                c2.cost = costfunc(y_hat_2, y_test)
                if c2.cost < bestsol.cost:
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