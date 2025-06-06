#!/usr/bin/env python
# Created by "Thieu" at 00:50, 02/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from scipy.stats import uniform, randint
from pathlib import Path
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from skopt import space as skace
from data_util import get_rt_iot2022
from helper import run_grid_search, run_random_search, run_bayes_search, run_optuna, run_meta_sklearn
from metasklearn import IntegerVar, StringVar, FloatVar, CategoricalVar, DataTransformer
from config import Config


def suggest_svc_params(trial):
    return {
        "C": trial.suggest_float("C", 1.0, 50.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid'])
    }

def suggest_rf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 6, 10),  # exclude None to avoid dtype issue
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 16)
    }

def suggest_mlp_params(trial):
    return {
        "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(40, 20), (50, 30), (60, 15)]),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "solver": trial.suggest_categorical("solver", ['lbfgs', 'adam']),
        "learning_rate": trial.suggest_categorical("learning_rate", ['constant', 'adaptive']),
        "alpha": trial.suggest_float("alpha", 1e-3, 0.2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024])
    }


if __name__ == "__main__":

    models = {
        "SVC": SVC(random_state=Config.SEED,),
        "RF": RandomForestClassifier(random_state=Config.SEED),
        "MLP": MLPClassifier(max_iter=1000, random_state=Config.SEED, early_stopping=True)
    }
    HIDDEN_SET = [(40, 20), (50, 30), (60, 15)]
    ALPHA_SET = [0.001, 0.01, 0.1]
    BATCH_SIZE_SET = [256, 512, 1024]

    param_grids = {
        "SVC": {
            "C": [1., 5., 10., 20., 50.],
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
        },
        "RF": {
            "n_estimators": [50, 100, 200, 300, 400],
            "max_depth": [None, 6, 7, 8, 9],
            "min_samples_split": [5, 10, 15],
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
        "SVC": {
            "C": uniform(1, 50),
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
        },
        "RF": {
            "n_estimators": randint(50, 401),
            "max_depth": [None, 6, 7, 8, 9],
            "min_samples_split": randint(5, 16)
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
        "SVC": {
            "C": skace.Real(1e0, 5e1, prior='log-uniform'),  # log scale between 1 and 50
            "kernel": skace.Categorical(['linear', 'poly', 'rbf', 'sigmoid'])  # avoid 'precomputed'
        },
        "RF": {
            "n_estimators": skace.Integer(50, 400),
            "max_depth": skace.Integer(6, 10),  # exclude None to avoid type mismatch in skopt
            "min_samples_split": skace.Integer(2, 16),
        },
        "MLP": {
            "hidden_layer_sizes": skace.Categorical([80, 70, 60]),  # HIDDEN_SET
            "activation": skace.Categorical(["relu", "tanh"]),
            "solver": skace.Categorical(['lbfgs', 'adam']),
            "learning_rate": skace.Categorical(['constant', 'adaptive']),
            "alpha": skace.Real(1e-3, 0.2, prior="log-uniform"),
            "batch_size": skace.Categorical(BATCH_SIZE_SET)
        }
    }

    param_bounds = {
        "SVC": [
            FloatVar(lb=0.01, ub=50., name="C"),
            StringVar(valid_sets=('linear', 'poly', 'rbf', 'sigmoid'), name="kernel"),
        ],
        "RF": [
            IntegerVar(lb=50, ub=400, name="n_estimators"),
            IntegerVar(lb=6, ub=10, name="max_depth"),  # exclude None to avoid type mismatch in skopt
            IntegerVar(lb=2, ub=16, name="min_samples_split"),
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
        "SVC": suggest_svc_params,
        "RF": suggest_rf_params,
        "MLP": suggest_mlp_params
    }

    ## Load data object
    X_train, X_test, y_train, y_test = get_rt_iot2022()

    ## Scaling dataset
    dt = DataTransformer(scaling_methods=("minmax",))
    X_train_scaled = dt.fit_transform(X_train)
    X_test_scaled = dt.transform(X_test)

    data = (X_train_scaled, y_train, X_test_scaled, y_test)
    Path(f"{Config.PATH_SAVE}/{Config.DATA_02}").mkdir(parents=True, exist_ok=True)

    # Run hyperparameter tuning for each model
    results = []
    for idx, (model_name, model) in enumerate(models.items()):
        res1 = run_grid_search(data, task_type="classification", model=model,
                               model_name=model_name, param_grid=param_grids[model_name], scoring='accuracy')
        res2 = run_random_search(data, task_type="classification", model=model,
                                 model_name=model_name, param_dist=param_dists[model_name], scoring='accuracy')
        res3 = run_bayes_search(data, task_type="classification", model=model, model_name=model_name,
                                param_space=param_spaces[model_name], scoring='accuracy')
        res4 = run_optuna(data, task_type="classification", model=model, model_name=model_name,
                          param_func=param_funcs[model_name], scoring='AS', direction="maximize")
        res5 = run_meta_sklearn(data, task_type="classification", model=model, model_name=model_name,
                                param_bounds=param_bounds[model_name], scoring='F1S', optim="RW_GWO",
                                optim_params={"epoch": Config.EPOCH, "pop_size": Config.POP_SIZE, "name": "RW-GWO"})
        res6 = run_meta_sklearn(data, task_type="classification", model=model, model_name=model_name,
                                param_bounds=param_bounds[model_name], scoring='F1S', optim="OriginalINFO",
                                optim_params={"epoch": Config.EPOCH, "pop_size": Config.POP_SIZE, "name": "INFO"})
        res7 = run_meta_sklearn(data, task_type="classification", model=model, model_name=model_name,
                                param_bounds=param_bounds[model_name], scoring='F1S', optim="OriginalSHADE",
                                optim_params={"epoch": Config.EPOCH, "pop_size": Config.POP_SIZE, "name": "SHADE"})
        res8 = run_meta_sklearn(data, task_type="classification", model=model, model_name=model_name,
                                param_bounds=param_bounds[model_name], scoring='F1S', optim="OriginalARO",
                                optim_params={"epoch": Config.EPOCH, "pop_size": Config.POP_SIZE, "name": "ARO"})
        results += [res1, res2, res3, res4, res5, res6, res7, res8]
        print(f"Done with model: {model_name}.")
    df_result = pd.DataFrame(results)  # Each row is a summary of metrics for a model/seed
    df_result.to_csv(f"{Config.PATH_SAVE}/{Config.DATA_02}/{Config.RESULT_METRICS}", index=False, header=True)
    print(f"Done with data: {Config.DATA_02}.")
