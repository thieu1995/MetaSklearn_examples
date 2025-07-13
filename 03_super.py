#!/usr/bin/env python
# Created by "Thieu" at 22:48, 06/06/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from scipy.stats import uniform, randint
from pathlib import Path
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.neural_network import MLPRegressor
from skopt import space as skace
from data_util import get_superconductivty
from helper import run_grid_search, run_random_search, run_bayes_search, run_optuna, run_meta_sklearn
from metasklearn import IntegerVar, StringVar, FloatVar, CategoricalVar, DataTransformer, SequenceVar
from config import Config


def suggest_svc_params(trial):
    return {
        "C": trial.suggest_float("C", 1., 20.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid']),
        "gamma": trial.suggest_categorical("gamma", ['scale', 'auto', 1e-3, 1e-4, 1e-5]),
        "coef0": trial.suggest_float("coef0", 0.0, 1.0)
    }

def suggest_rf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 60, 100),
        "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy']),
        "max_depth": trial.suggest_int("max_depth", 4, 7),  # exclude None to avoid dtype issue
        "min_samples_split": trial.suggest_int("min_samples_split", 3, 7),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 4),
        "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2', 4, 5, 0.15, 0.25])
    }

def suggest_mlp_params(trial):
    return {
        "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(70, 30), (50, 20), (60, ), (50, )]),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic", "identity"]),
        "solver": trial.suggest_categorical("solver", ['lbfgs', 'adam']),
        "alpha": trial.suggest_float("alpha", 1e-3, 0.2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "learning_rate": trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive']),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 0.001, 0.1, log=True)  # initial learning rate
    }

def run_trial(model_name, model_object, data, params, epoch, pop_size, path_save):
    param_grid, param_dist, param_space, param_func, param_bound = params
    res1 = run_grid_search(data, task_type="regression", model=model_object,
                           model_name=model_name, param_grid=param_grid, scoring='neg_mean_squared_error')
    res2 = run_random_search(data, task_type="regression", model=model_object,
                             model_name=model_name, param_dist=param_dist, scoring='neg_mean_squared_error')
    res3 = run_bayes_search(data, task_type="regression", model=model_object, model_name=model_name,
                            param_space=param_space, scoring='neg_mean_squared_error')
    res4 = run_optuna(data, task_type="regression", model=model_object, model_name=model_name,
                      param_func=param_func, scoring='MSE', direction="minimize")
    res5 = run_meta_sklearn(data, task_type="regression", model=model_object, model_name=model_name,
                            param_bounds=param_bound, scoring='MSE', optim="RW_GWO",
                            optim_params={"epoch": epoch, "pop_size": pop_size, "name": "RW-GWO"})
    res6 = run_meta_sklearn(data, task_type="regression", model=model_object, model_name=model_name,
                            param_bounds=param_bound, scoring='MSE', optim="OriginalINFO",
                            optim_params={"epoch": epoch, "pop_size": pop_size, "name": "INFO"})
    res7 = run_meta_sklearn(data, task_type="regression", model=model_object, model_name=model_name,
                            param_bounds=param_bound, scoring='MSE', optim="OriginalSHADE",
                            optim_params={"epoch": epoch, "pop_size": pop_size, "name": "SHADE"})
    res8 = run_meta_sklearn(data, task_type="regression", model=model_object, model_name=model_name,
                            param_bounds=param_bound, scoring='MSE', optim="OriginalARO",
                            optim_params={"epoch": epoch, "pop_size": pop_size, "name": "ARO"})
    print(f"Done with model: {model_name}.")
    df_result = pd.DataFrame([res1, res2, res3, res4, res5, res6, res7, res8])
    df_result.to_csv(path_save, index=False, header=True)


if __name__ == "__main__":
    ## Configurations
    HIDDEN_SET = [(70, 30), (50, 20), (60, ), (50, )]
    ALPHA_SET = [0.001, 0.01, 0.1]
    BATCH_SIZE_SET = [32, 64, 128]

    param_grids = {
        "SVC": {
            "C": [1., 5., 10., 15., 20.],
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "gamma": ['scale', 'auto', 1e-3, 1e-4, 1e-5],
            "coef0": [0., 0.1, 0.25, 0.5, 1.0]
        },
        "RF": {
            "n_estimators": [60, 70, 80, 90, 100],
            "criterion": ['gini', 'entropy'],
            "max_depth": [None, 4, 5, 6, 7],
            "min_samples_split": [3, 5, 7],
            "min_samples_leaf": [2, 3, 4],
            "max_features": ['sqrt', 'log2', 4, 5, 0.15, 0.25]
        },
        "MLP": {
            "hidden_layer_sizes": HIDDEN_SET,
            "activation": ["relu", "tanh", "logistic", "identity"],
            "solver": ['lbfgs', 'adam'],
            "alpha": ALPHA_SET,
            "batch_size": BATCH_SIZE_SET,
            "learning_rate": ['constant', 'invscaling', 'adaptive'],
            "learning_rate_init": [0.001, 0.01, 0.1]  # initial learning rate
        }
    }

    param_dists = {
        "SVC": {
            "C": uniform(1., 20.),
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "gamma": ['scale', 'auto', 1e-3, 1e-4, 1e-5],
            "coef0": uniform(0, 1)
        },
        "RF": {
            "n_estimators": randint(60, 100),
            "criterion": ['gini', 'entropy'],
            "max_depth": [None, 4, 5, 6, 7],
            "min_samples_split": randint(3, 7),
            "min_samples_leaf": randint(2, 4),
            "max_features": ['sqrt', 'log2', 4, 5, 0.15, 0.25]
        },
        "MLP": {
            "hidden_layer_sizes": HIDDEN_SET,
            "activation": ["relu", "tanh", "logistic", "identity"],
            "solver": ['lbfgs', 'adam'],
            "alpha": uniform(0.001, 0.2),  # continuous sampling from 0.001 to 0.201
            "batch_size": BATCH_SIZE_SET,
            "learning_rate": ['constant', 'invscaling', 'adaptive'],
            "learning_rate_init": uniform(0.001, 0.1)  # continuous sampling from 0.001 to 0.101
        }
    }

    param_spaces = {
        "SVC": {
            "C": skace.Real(1., 20., prior='uniform'),  # scale between 1 and 50
            "kernel": skace.Categorical(['linear', 'poly', 'rbf', 'sigmoid']),  # avoid 'precomputed'
            "gamma": skace.Categorical(['scale', 'auto', 1e-3, 1e-4, 1e-5]),
            "coef0": skace.Real(0.0, 1.0, prior='uniform')  # uniform scale between 0 and 1
        },
        "RF": {
            "n_estimators": skace.Integer(60, 100),
            "criterion": skace.Categorical(['gini', 'entropy']),
            "max_depth": skace.Integer(4, 7),  # exclude None to avoid type mismatch in skopt
            "min_samples_split": skace.Integer(3, 7),
            "min_samples_leaf": skace.Integer(2, 4),  # minimum leaf size
            "max_features": skace.Categorical(['sqrt', 'log2', 4, 5, 0.15, 0.25])  # categorical features
        },
        "MLP": {
            "hidden_layer_sizes": skace.Categorical([60, 50, 40]),  # HIDDEN_SET
            "activation": skace.Categorical(["relu", "tanh", "logistic", "identity"]),
            "solver": skace.Categorical(['lbfgs', 'adam']),
            "alpha": skace.Real(1e-3, 0.2, prior="uniform"),
            "batch_size": skace.Categorical(BATCH_SIZE_SET),
            "learning_rate": skace.Categorical(['constant', 'invscaling', 'adaptive']),
            "learning_rate_init": skace.Real(1e-3, 0.1, prior="log-uniform")  # log scale for initial learning rate
        }
    }

    param_bounds = {
        "SVC": [
            FloatVar(lb=1., ub=20., name="C"),
            StringVar(valid_sets=('linear', 'poly', 'rbf', 'sigmoid'), name="kernel"),
            CategoricalVar(valid_sets=('scale', 'auto', 1e-3, 1e-4, 1e-5), name="gamma"),
            FloatVar(lb=0.0, ub=1.0, name="coef0")
        ],
        "RF": [
            IntegerVar(lb=60, ub=100, name="n_estimators"),
            StringVar(valid_sets=('gini', 'entropy'), name="criterion"),
            CategoricalVar(valid_sets=(None, 4, 5, 6, 7), name="max_depth"),
            IntegerVar(lb=3, ub=7, name="min_samples_split"),
            IntegerVar(lb=2, ub=4, name="min_samples_leaf"),
            CategoricalVar(valid_sets=('sqrt', 'log2', 4, 5, 0.15, 0.25), name="max_features")
        ],
        "MLP": [
            SequenceVar(valid_sets=(HIDDEN_SET), name="hidden_layer_sizes"),
            StringVar(valid_sets=("relu", "tanh", "logistic", "identity"), name="activation"),
            StringVar(valid_sets=('lbfgs', 'adam'), name="solver"),
            FloatVar(lb=0.001, ub=0.2, name="alpha"),
            CategoricalVar(valid_sets=BATCH_SIZE_SET, name="batch_size"),
            StringVar(valid_sets=('constant', 'invscaling', 'adaptive'), name="learning_rate"),
            FloatVar(lb=0.001, ub=0.1, name="learning_rate_init")  # initial learning rate
        ]
    }

    param_funcs = {
        "SVC": suggest_svc_params,
        "RF": suggest_rf_params,
        "MLP": suggest_mlp_params
    }

    ## Load data object
    X_train, X_test, y_train, y_test = get_superconductivty()

    ## Scaling dataset
    dt = DataTransformer(scaling_methods=("minmax",))
    X_train_scaled = dt.fit_transform(X_train)
    X_test_scaled = dt.transform(X_test)
    
    dt_y = DataTransformer(scaling_methods=("minmax",))
    y_train_scaled = dt_y.fit_transform(y_train)
    y_test_scaled = dt_y.transform(y_test)

    data = (X_train_scaled, y_train_scaled.ravel(), X_test_scaled, y_test_scaled.ravel())
    Path(f"{Config.PATH_SAVE}/{Config.DATA_03}").mkdir(parents=True, exist_ok=True)

    LIST_MODELS = [
        {"name": "SVC", "object": SVR(),},
        {"name": "RF", "object": RandomForestRegressor(random_state=Config.SEED),},
        {"name": "MLP", "object": MLPRegressor(max_iter=1000, random_state=Config.SEED, early_stopping=True)},
    ]

    # Run parallel
    with ProcessPoolExecutor(max_workers=Config.N_WORKERS) as executor:
        futures = []
        for model in LIST_MODELS:
            pathsave = f"{Config.PATH_SAVE}/{Config.DATA_03}/{model['name']}_results.csv"
            params = [param_grids[model["name"]], param_dists[model['name']],
                      param_spaces[model['name']], param_funcs[model['name']], param_bounds[model['name']]]
            futures.append(executor.submit(run_trial, model['name'], model['object'],
                                           data, params, Config.EPOCH, Config.POP_SIZE, pathsave))

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An exception occurred: {e}")
    print(f"Done with data: {Config.DATA_03}.")
