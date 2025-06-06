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
from sklearn.neural_network import MLPRegressor
from skopt import space as skace
from data_util import get_superconductivty
from helper import run_grid_search, run_random_search, run_bayes_search, run_optuna, run_meta_sklearn
from metasklearn import IntegerVar, StringVar, FloatVar, CategoricalVar, DataTransformer
from config import Config


def suggest_svc_params(trial):
    return {
        "C": trial.suggest_float("C", 1.0, 20.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid'])
    }

def suggest_rf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 120),
        "max_depth": trial.suggest_int("max_depth", 4, 8),  # exclude None to avoid dtype issue
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 7)
    }

def suggest_mlp_params(trial):
    return {
        "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [50, 40, 30]),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "solver": trial.suggest_categorical("solver", ['lbfgs', 'adam']),
        "learning_rate": trial.suggest_categorical("learning_rate", ['constant', 'adaptive']),
        "alpha": trial.suggest_float("alpha", 1e-3, 0.2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128])
    }


if __name__ == "__main__":

    models = {
        "SVM": SVR(),
        "RF": RandomForestRegressor(random_state=Config.SEED),
        "MLP": MLPRegressor(max_iter=1000, random_state=Config.SEED, early_stopping=True)
    }
    HIDDEN_SET = [50, 40, 30]
    ALPHA_SET = [0.001, 0.01, 0.1]
    BATCH_SIZE_SET = [32, 64, 128]

    param_grids = {
        "SVM": {
            "C": [1., 5., 10., 15., 20.],
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
        },
        "RF": {
            "n_estimators": [50, 70, 90, 100, 120],
            "max_depth": [None, 4, 5, 6, 7],
            "min_samples_split": [3, 5, 7],
        },
        "MLP": {
            "hidden_layer_sizes": HIDDEN_SET,
            "activation": ["relu", "tanh"],
            "solver": ['lbfgs', 'adam'],
            "learning_rate": ['constant', 'adaptive'],
            "alpha": ALPHA_SET,
            "batch_size": BATCH_SIZE_SET,
        }
    }

    param_dists = {
        "SVM": {
            "C": uniform(1, 20),
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
        },
        "RF": {
            "n_estimators": randint(50, 120),
            "max_depth": [None, 4, 5, 6, 7],
            "min_samples_split": randint(3, 7)
        },
        "MLP": {
            "hidden_layer_sizes": HIDDEN_SET,
            "activation": ["relu", "tanh"],
            "solver": ['lbfgs', 'adam'],
            "learning_rate": ['constant', 'adaptive'],
            "alpha": uniform(0.001, 0.2),  # continuous sampling from 0.001 to 0.201
            "batch_size": BATCH_SIZE_SET
        }
    }

    param_spaces = {
        "SVM": {
            "C": skace.Real(1e0, 2e1, prior='log-uniform'),  # log scale between 1 and 50
            "kernel": skace.Categorical(['linear', 'poly', 'rbf', 'sigmoid'])  # avoid 'precomputed'
        },
        "RF": {
            "n_estimators": skace.Integer(50, 120),
            "max_depth": skace.Integer(4, 7),  # exclude None to avoid type mismatch in skopt
            "min_samples_split": skace.Integer(2, 8),
        },
        "MLP": {
            "hidden_layer_sizes": skace.Categorical(HIDDEN_SET),  # HIDDEN_SET
            "activation": skace.Categorical(["relu", "tanh"]),
            "solver": skace.Categorical(['lbfgs', 'adam']),
            "learning_rate": skace.Categorical(['constant', 'adaptive']),
            "alpha": skace.Real(1e-3, 0.2, prior="log-uniform"),
            "batch_size": skace.Categorical(BATCH_SIZE_SET)
        }
    }

    param_bounds = {
        "SVM": [
            FloatVar(lb=0.01, ub=20., name="C"),
            StringVar(valid_sets=('linear', 'poly', 'rbf', 'sigmoid'), name="kernel"),
        ],
        "RF": [
            IntegerVar(lb=50, ub=120, name="n_estimators"),
            IntegerVar(lb=4, ub=8, name="max_depth"),  # exclude None to avoid type mismatch in skopt
            IntegerVar(lb=2, ub=7, name="min_samples_split"),
        ],
        "MLP": [
            CategoricalVar(valid_sets=(HIDDEN_SET), name="hidden_layer_sizes"),
            StringVar(valid_sets=("relu", "tanh"), name="activation"),
            StringVar(valid_sets=('lbfgs', 'adam'), name="solver"),
            StringVar(valid_sets=('constant', 'adaptive'), name="learning_rate"),
            FloatVar(lb=0.001, ub=0.2, name="alpha"),
            CategoricalVar(valid_sets=BATCH_SIZE_SET, name="batch_size")
        ]
    }

    param_funcs = {
        "SVM": suggest_svc_params,
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

    data = (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    Path(f"{Config.PATH_SAVE}/{Config.DATA_03}").mkdir(parents=True, exist_ok=True)

    # Run hyperparameter tuning for each model
    results = []
    for idx, (model_name, model) in enumerate(models.items()):
        res1 = run_grid_search(data, task_type="regression", model=model,
                               model_name=model_name, param_grid=param_grids[model_name], scoring='neg_mean_squared_error')
        res2 = run_random_search(data, task_type="regression", model=model,
                                 model_name=model_name, param_dist=param_dists[model_name], scoring='neg_mean_squared_error')
        res3 = run_bayes_search(data, task_type="regression", model=model, model_name=model_name,
                                param_space=param_spaces[model_name], scoring='neg_mean_squared_error')
        res4 = run_optuna(data, task_type="regression", model=model, model_name=model_name,
                          param_func=param_funcs[model_name], scoring='MSE', direction="maximize")
        res5 = run_meta_sklearn(data, task_type="regression", model=model, model_name=model_name,
                                param_bounds=param_bounds[model_name], scoring='MSE', optim="RW_GWO",
                                optim_params={"epoch": Config.EPOCH, "pop_size": Config.POP_SIZE, "name": "RW-GWO"})
        res6 = run_meta_sklearn(data, task_type="regression", model=model, model_name=model_name,
                                param_bounds=param_bounds[model_name], scoring='MSE', optim="OriginalINFO",
                                optim_params={"epoch": Config.EPOCH, "pop_size": Config.POP_SIZE, "name": "INFO"})
        res7 = run_meta_sklearn(data, task_type="regression", model=model, model_name=model_name,
                                param_bounds=param_bounds[model_name], scoring='MSE', optim="OriginalSHADE",
                                optim_params={"epoch": Config.EPOCH, "pop_size": Config.POP_SIZE, "name": "SHADE"})
        res8 = run_meta_sklearn(data, task_type="regression", model=model, model_name=model_name,
                                param_bounds=param_bounds[model_name], scoring='MSE', optim="OriginalARO",
                                optim_params={"epoch": Config.EPOCH, "pop_size": Config.POP_SIZE, "name": "ARO"})
        results += [res1, res2, res3, res4, res5, res6, res7, res8]
        print(f"Done with model: {model_name}.")
    df_result = pd.DataFrame(results)  # Each row is a summary of metrics for a model/seed
    df_result.to_csv(f"{Config.PATH_SAVE}/{Config.DATA_03}/{Config.RESULT_METRICS}", index=False, header=True)
    print(f"Done with data: {Config.DATA_03}.")
