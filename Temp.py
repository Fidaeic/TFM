# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:30:18 2020

@author: Propietario
"""
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
import wrangler
import recon
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import sherpa.algorithms.bayesian_optimization as bayesian_optimization
import optuna

df_flow = pd.read_csv("Data/flow.csv", parse_dates=[3], sep=',', header = None).drop([0, 1, 2], axis=1)
df_flow, almac, nul, bloques = wrangler.data_wrangler(df_flow)
df_mediana, time = recon.recon_mean(df_flow, nul)
X_train, X_test, y_train, y_test, X, y, y_for_test= wrangler.shifting(df_mediana, estac=96, horizon=96)

import sherpa
'''
parameters = [sherpa.Discrete('num_units', [50, 200])]
alg = sherpa.algorithms.RandomSearch(max_num_trials=50)

study = sherpa.Study(parameters=parameters,
                     algorithm=alg,
                     lower_is_better=True,
                     disable_dashboard=True)

for trial in study:
    model = Sequential()
    model.add(Dense(units=trial.parameters['num_units'],
                    activation='relu', input_dim=96))
    model.add(Dense(units=1, activation='relu'))
    model.compile(loss='mse',
              optimizer='adam')

    model.fit(X_train, y_train, epochs=5, batch_size=32,
              callbacks=[study.keras_callback(trial, objective_name='loss')])
    study.finalize(trial)
'''



# =============================================================================
# RF
# =============================================================================

parameters = [sherpa.Discrete('n_estimators', [2, 50]),
              sherpa.Choice('criterion', ['gini', 'entropy']),
              sherpa.Continuous('max_features', [0.1, 0.9])]

algorithm = sherpa.algorithms.PopulationBasedTraining(population_size=100,
                                                      num_generations=5,
                                                      perturbation_factors=(0.8, 1.2),
                                                      parameter_range={'n_estimators': [1, 50]})
study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     lower_is_better=False,
                     disable_dashboard=True)

for trial in study:
    generation = trial.parameters['generation']
    load_from = trial.parameters['load_from']
    
    print("-"*100)
    print("Generation {}".format(generation))

    print("Trial ", trial.id, " with parameters ", trial.parameters)
    reg = RandomForestRegressor(criterion=trial.parameters['criterion'],
                             max_features=trial.parameters['max_features'],
                             n_estimators=trial.parameters['n_estimators'],
                             random_state=0)

    scores = cross_val_score(reg, X_train, y_train, cv=5)
    print("Score: ", scores.mean())
    study.add_observation(trial, iteration=1, objective=scores.mean())
    study.finalize(trial)
print(study.get_best_result())


def objective(trial):
    
    reg_name = trial.suggest_categorical("regressor", ["RandomForest"])
    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 50, log=True)
    n_est = trial.suggest_int("rf_n_est", 2, 100)
    reg_obj = RandomForestRegressor(
        max_depth=rf_max_depth, n_estimators=n_est
    )

    score = cross_val_score(reg_obj, X, y, n_jobs=-1, cv=5)
    accuracy = score.mean()
    return accuracy, score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print(study.best_trial)
    




# =============================================================================
# PRUEBA
# =============================================================================

import sherpa
class Opti(sherpa.algorithms.Algorithm):
    def get_suggestion(self, parameters, results, lower_is_better):
            """
            Create a new parameter value as a random mixture of some of the best
            trials and sampling from original distribution.

            Returns:
                dict: parameter values dictionary
            """
            # Choose 2 of the top trials and get their parameter values
            trial_1_params = self._get_candidate(parameters, results, lower_is_better)
            trial_2_params = self._get_candidate(parameters, results, lower_is_better)
            params_values_for_next_trial = {}
            for param_name in trial_1_params.keys():
                param_origin = np.random.randint(3)  # randomly choose where to get the value from
                if param_origin == 1:
                    params_values_for_next_trial[param_name] = trial_1_params[param_name]
                elif param_origin == 2:
                    params_values_for_next_trial[param_name] = trial_2_params[param_name]
                else:
                    for parameter_object in parameters:
                        if param_name == parameter_object.name:
                            params_values_for_next_trial[param_name] = parameter_object.sample()
            return params_values_for_next_trial
    
    def _get_candidate(self, parameters, results, lower_is_better, min_candidates=10):
        """
        Samples candidates parameters from the top 33% of population.
        Returns:
            dict: parameter dictionary.
        """
        if results.shape[0] > 0: # In case this is the first trial
            population = results.loc[results['Status'] != 'INTERMEDIATE', :]  # select only completed trials
        else: # In case this is the first trial
            population = None
        if population is None or population.shape[0] < min_candidates: # Generate random values
            for parameter_object in parameters:
                trial_param_values[parameter_object.name] = parameter_object.sample()
                    return trial_param_values
        population = population.sort_values(by='Objective', ascending=lower_is_better)
        idx = numpy.random.randint(low=0, high=population.shape[0]//3)  # pick randomly among top 33%
        trial_all_values = population.iloc[idx].to_dict()  # extract the trial values on results table
        trial_param_values = {param.name: d[param.name] for param in parameters} # Select only parameter values
        return trial_param_values