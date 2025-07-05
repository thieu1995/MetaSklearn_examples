#!/usr/bin/env python
# Created by "Thieu" at 00:50, 02/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from scipy.stats import uniform, randint
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from skopt import space as skace
from data_util import get_bank_marketing
from helper import run_grid_search, run_random_search, run_bayes_search, run_optuna, run_meta_sklearn
from metasklearn import IntegerVar, StringVar, FloatVar, CategoricalVar, DataTransformer, SequenceVar
from config import Config


def suggest_svc_params(trial):
    return {
        "C": trial.suggest_float("C", 1.0, 50.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid']),
        "gamma": trial.suggest_categorical("gamma", ['scale', 'auto', 1e-3, 1e-4, 1e-5]),
        "coef0": trial.suggest_float("coef0", 0.0, 1.0)
    }

def suggest_rf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy']),
        "max_depth": trial.suggest_int("max_depth", 5, 8),  # exclude None to avoid dtype issue
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 5),
        "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2', 2, 3, 0.15, 0.25])
    }

def suggest_mlp_params(trial):
    return {
        "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(30, 10), (50, 20), (40, ), (30, )]),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic", "identity"]),
        "solver": trial.suggest_categorical("solver", ['lbfgs', 'adam']),
        "alpha": trial.suggest_float("alpha", 1e-3, 0.2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "learning_rate": trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive']),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 0.001, 0.1, log=True)  # initial learning rate
    }

def run_trial(model_name, model_object, data, params, epoch, pop_size, path_save):
    results = []
    param_grid, param_dist, param_space, param_func, param_bound = params
    res1 = run_grid_search(data, task_type="classification", model=model_object,
                           model_name=model_name, param_grid=param_grid, scoring='accuracy')
    res2 = run_random_search(data, task_type="classification", model=model_object,
                             model_name=model_name, param_dist=param_dist, scoring='accuracy')
    res3 = run_bayes_search(data, task_type="classification", model=model_object, model_name=model_name,
                            param_space=param_space, scoring='accuracy')
    res4 = run_optuna(data, task_type="classification", model=model_object, model_name=model_name,
                      param_func=param_func, scoring='AS', direction="maximize")
    res5 = run_meta_sklearn(data, task_type="classification", model=model_object, model_name=model_name,
                            param_bounds=param_bound, scoring='F1S', optim="RW_GWO",
                            optim_params={"epoch": epoch, "pop_size": pop_size, "name": "RW-GWO"})
    res6 = run_meta_sklearn(data, task_type="classification", model=model_object, model_name=model_name,
                            param_bounds=param_bound, scoring='F1S', optim="OriginalINFO",
                            optim_params={"epoch": epoch, "pop_size": pop_size, "name": "INFO"})
    res7 = run_meta_sklearn(data, task_type="classification", model=model_object, model_name=model_name,
                            param_bounds=param_bound, scoring='F1S', optim="OriginalSHADE",
                            optim_params={"epoch": epoch, "pop_size": pop_size, "name": "SHADE"})
    res8 = run_meta_sklearn(data, task_type="classification", model=model_object, model_name=model_name,
                            param_bounds=param_bound, scoring='F1S', optim="OriginalARO",
                            optim_params={"epoch": epoch, "pop_size": pop_size, "name": "ARO"})
    print(f"Done with model: {model_name}.")
    df_result = pd.DataFrame([res1, res2, res3, res4, res5, res6, res7, res8])
    df_result.to_csv(path_save, index=False, header=True)

if __name__ == "__main__":
    ## Configurations
    HIDDEN_SET = [(30, 10), (50, 20), (40, ), (30, )]
    ALPHA_SET = [0.001, 0.01, 0.1, 0.2]
    BATCH_SIZE_SET = [32, 64, 128]

    param_grids = {
        "SVC": {
            "C": [1., 5., 10., 20., 50.],
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "gamma": ['scale', 'auto', 1e-3, 1e-4, 1e-5],
            "coef0": [0., 0.1, 0.25, 0.5, 1.0]
        },
        "RF": {
            "n_estimators": [50, 70, 90, 120, 150],
            "criterion": ['gini', 'entropy'],
            "max_depth": [None, 5, 6, 7, 8],
            "min_samples_split": [5, 7, 10],
            "min_samples_leaf": [2, 3, 4],
            "max_features": ['sqrt', 'log2', 2, 3, 0.15, 0.25]
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
            "C": uniform(1, 50),
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "gamma": ['scale', 'auto', 1e-3, 1e-4, 1e-5],
            "coef0": uniform(0, 1)
        },
        "RF": {
            "n_estimators": randint(50, 150),
            "criterion": ['gini', 'entropy'],
            "max_depth": [None, 5, 6, 7, 8],
            "min_samples_split": randint(5, 10),
            "min_samples_leaf": randint(2, 5),
            "max_features": ['sqrt', 'log2', 2, 3, 0.15, 0.25]
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
            "C": skace.Real(1e0, 5e1, prior='log-uniform'),  # log scale between 1 and 50
            "kernel": skace.Categorical(['linear', 'poly', 'rbf', 'sigmoid']),  # avoid 'precomputed'
            "gamma": skace.Categorical(['scale', 'auto', 1e-3, 1e-4, 1e-5]),
            "coef0": skace.Real(0.0, 1.0, prior='uniform')  # uniform scale between 0 and 1
        },
        "RF": {
            "n_estimators": skace.Integer(50, 150),
            "criterion": skace.Categorical(['gini', 'entropy']),
            "max_depth": skace.Integer(5, 8),  # exclude None to avoid type mismatch in skopt
            "min_samples_split": skace.Integer(5, 10),
            "min_samples_leaf": skace.Integer(2, 5),  # minimum leaf size
            "max_features": skace.Categorical(['sqrt', 'log2', 2, 3, 0.15, 0.25])  # categorical features
        },
        "MLP": {
            "hidden_layer_sizes": skace.Categorical([50, 40, 30]),  # HIDDEN_SET
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
            FloatVar(lb=0.01, ub=50., name="C"),
            StringVar(valid_sets=('linear', 'poly', 'rbf', 'sigmoid'), name="kernel"),
            CategoricalVar(valid_sets=('scale', 'auto', 1e-3, 1e-4, 1e-5), name="gamma"),
            FloatVar(lb=0.0, ub=1.0, name="coef0")
        ],
        "RF": [
            IntegerVar(lb=50, ub=150, name="n_estimators"),
            StringVar(valid_sets=('gini', 'entropy'), name="criterion"),
            CategoricalVar(valid_sets=(None, 5, 6, 7, 8), name="max_depth"),
            IntegerVar(lb=2, ub=5, name="min_samples_split"),
            IntegerVar(lb=1, ub=5, name="min_samples_leaf"),
            CategoricalVar(valid_sets=('sqrt', 'log2', 2, 3, 0.15, 0.25), name="max_features")
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
    X_train, X_test, y_train, y_test = get_bank_marketing()

    ## Scaling dataset
    dt = DataTransformer(scaling_methods=("minmax",))
    X_train_scaled = dt.fit_transform(X_train)
    X_test_scaled = dt.transform(X_test)

    data = (X_train_scaled, y_train, X_test_scaled, y_test)
    Path(f"{Config.PATH_SAVE}/{Config.DATA_01}").mkdir(parents=True, exist_ok=True)

    LIST_MODELS = [
        {"name": "SVC", "object": SVC(random_state=Config.SEED,)},
        {"name": "RF", "object": RandomForestClassifier(random_state=Config.SEED)},
        {"name": "MLP", "object": MLPClassifier(max_iter=1000, random_state=Config.SEED, early_stopping=True)},
    ]

    # Run trials in parallel for all models and seeds
    all_epoch_losses = []
    all_results = []

    with ProcessPoolExecutor(max_workers=Config.N_WORKERS) as executor:
        futures = []
        for model_name, model_object in LIST_MODELS:
            pathsave = f"{Config.PATH_SAVE}/{Config.DATA_01}/{model_name}_results.csv"
            params = [param_grids[model_name], param_dists[model_name],
                      param_spaces[model_name], param_funcs[model_name], param_bounds[model_name]]
            futures.append(executor.submit(run_trial, model_name, model_object,
                                           data, params, Config.EPOCH, Config.POP_SIZE, pathsave))

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An exception occurred: {e}")
    print(f"Done with data: {Config.DATA_01}.")
