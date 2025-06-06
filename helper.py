#!/usr/bin/env python
# Created by "Thieu" at 16:19, 27/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import time
from config import Config
from permetrics import RegressionMetric, ClassificationMetric
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
import optuna
from metasklearn import MetaSearchCV


def run_grid_search(data, task_type="classification", model=None, model_name=None,
                    param_grid=None, scoring='accuracy'):
    time_start = time.perf_counter()
    X_train, y_train, X_test, y_test, = data
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, verbose=0)
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    if task_type == "regression":
        mt = RegressionMetric(y_test, y_pred)
        res = mt.get_metrics_by_list_names(list_metric_names=Config.LIST_METRIC_REG)
    else:
        mt = ClassificationMetric(y_test, y_pred)
        res = mt.get_metrics_by_list_names(list_metric_names=Config.LIST_METRIC_CLS)
    time_end = time.perf_counter() - time_start
    res = {"method": "GridSearchCV", "model": model_name, **res,
           "run_time": time_end, "best_params": grid_search.best_params_,
           "best_estimator": grid_search.best_estimator_}
    print(f"GridSearchCV completed in {time_end:.2f} seconds for {model_name}.")
    return res


def run_random_search(data, task_type="classification", model=None, model_name=None,
                      param_dist=None, scoring='accuracy'):
    time_start = time.perf_counter()
    X_train, y_train, X_test, y_test, = data
    random_search = RandomizedSearchCV(model, param_dist, n_iter=Config.EPOCH,
                                       cv=3, scoring=scoring, verbose=0, random_state=Config.SEED)
    random_search.fit(X_train, y_train)

    y_pred = random_search.predict(X_test)
    if task_type == "regression":
        mt = RegressionMetric(y_test, y_pred)
        res = mt.get_metrics_by_list_names(list_metric_names=Config.LIST_METRIC_REG)
    else:
        mt = ClassificationMetric(y_test, y_pred)
        res = mt.get_metrics_by_list_names(list_metric_names=Config.LIST_METRIC_CLS)
    time_end = time.perf_counter() - time_start
    res = {"method": "RandomizedSearchCV", "model": model_name, **res,
           "run_time": time_end, "best_params": random_search.best_params_,
           "best_estimator": random_search.best_estimator_}
    print(f"RandomizedSearchCV completed in {time_end:.2f} seconds for {model_name}.")
    return res


def run_bayes_search(data, task_type="classification", model=None, model_name=None,
                     param_space=None, scoring='accuracy'):
    time_start = time.perf_counter()
    X_train, y_train, X_test, y_test, = data
    bayes_search = BayesSearchCV(model, param_space, n_iter=Config.EPOCH, cv=3, scoring=scoring,
                                 verbose=0, random_state=Config.SEED)
    bayes_search.fit(X_train, y_train)

    y_pred = bayes_search.predict(X_test)
    if task_type == "regression":
        mt = RegressionMetric(y_test, y_pred)
        res = mt.get_metrics_by_list_names(list_metric_names=Config.LIST_METRIC_REG)
    else:
        mt = ClassificationMetric(y_test, y_pred)
        res = mt.get_metrics_by_list_names(list_metric_names=Config.LIST_METRIC_CLS)
    time_end = time.perf_counter() - time_start
    res = {"method": "BayesSearchCV", "model": model_name, **res,
           "run_time": time_end, "best_params": bayes_search.best_params_,
           "best_estimator": bayes_search.best_estimator_}
    print(f"BayesSearchCV completed in {time_end:.2f} seconds for {model_name}.")
    return res


def run_optuna(data, task_type="classification", model=None, model_name=None,
               param_func=None, scoring='AS', direction="maximize"):
    time_start = time.perf_counter()
    X_train, y_train, X_test, y_test, = data

    def objective(trial):
        params = param_func(trial)
        clf = model.set_params(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if task_type == "classification":
            mt = ClassificationMetric(y_test, y_pred)
        else:
            mt = RegressionMetric(y_test, y_pred)
        return mt.get_metric_by_name(scoring)[scoring]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=Config.EPOCH)

    best_params = study.best_params
    best_estimator = model.set_params(**best_params).fit(X_train, y_train)
    y_pred = best_estimator.predict(X_test)
    if task_type == "regression":
        mt = RegressionMetric(y_test, y_pred)
        res = mt.get_metrics_by_list_names(list_metric_names=Config.LIST_METRIC_REG)
    else:
        mt = ClassificationMetric(y_test, y_pred)
        res = mt.get_metrics_by_list_names(list_metric_names=Config.LIST_METRIC_CLS)
    time_end = time.perf_counter() - time_start
    res = {"method": "Optuna", "model": model_name, **res,
           "run_time": time_end, "best_params": best_params,
           "best_estimator": best_estimator}
    print(f"Optuna optimization completed in {time_end:.2f} seconds for {model_name}.")
    return res


def run_meta_sklearn(data, task_type="classification", model=None, model_name=None, param_bounds=None,
                     scoring='F1S', optim="BaseGA", optim_params=None):
    time_start = time.perf_counter()
    X_train, y_train, X_test, y_test, = data

    # Initialize and fit MetaSearchCV
    searcher = MetaSearchCV(
        estimator=model,
        param_bounds=param_bounds,
        task_type=task_type,
        optim=optim,
        optim_params=optim_params,
        cv=3,
        scoring=scoring,  # or any custom scoring like "F1_macro"
        seed=Config.SEED,
        verbose=False
    )
    searcher.fit(X_train, y_train)

    y_pred = searcher.predict(X_test)
    if task_type == "regression":
        mt = RegressionMetric(y_test, y_pred)
        res = mt.get_metrics_by_list_names(list_metric_names=Config.LIST_METRIC_REG)
    else:
        mt = ClassificationMetric(y_test, y_pred)
        res = mt.get_metrics_by_list_names(list_metric_names=Config.LIST_METRIC_CLS)
    time_end = time.perf_counter() - time_start
    res = {"method": searcher.optim.name, "model": model_name, **res,
           "run_time": time_end, "best_params": searcher.best_params,
           "best_estimator": searcher.best_estimator}
    print(f"MetaSearchCV with {searcher.optim.name} completed in {time_end:.2f} seconds for {model_name}.")
    return res
