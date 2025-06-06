#!/usr/bin/env python
# Created by "Thieu" at 19:20, 10/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

class Config:

    DATA_01 = "cdc_diabetes_health"
    DATA_02 = "rt_iot2022.csv"
    DATA_03 = "superconductivty.csv"

    RESULT_LOSS = "df_loss.csv"
    RESULT_METRICS = "df_result.csv"
    RESULTS_PARAMS = "df_best_params.csv"

    SEED = 42

    EPOCH = 100
    POP_SIZE = 20
    CV = 5
    N_WORKERS = 10
    VERBOSE = False

    LIST_SEEDS = [10, 15, 21, 24, 27, 29, 30, 35, 40, 42]
    LIST_METRIC_CLS = ["AS", "PS", "RS", "F1S", "SS", "NPV"]
    LIST_METRIC_REG = ["MAE", "RMSE", "NNSE", "WI", "R", "KGE"]
    PATH_SAVE = "history_latest"

    PROBLEM_CLS = "classification"
    PROBLEM_REG = "regression"
    LOSS_CLS = "F1S"
    LOSS_REG = "MSE"

    # LIST_MODELS = [
    #     {"name": "BBO-RVFL", "class": "OriginalBBO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "SADE-RVFL", "class": "SADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "SHADE-RVFL", "class": "OriginalSHADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "LCO-RVFL", "class": "OriginalLCO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "INFO-RVFL", "class": "OriginalINFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "QLE-SCA-RVFL", "class": "QleSCA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "SHIO-RVFL", "class": "OriginalSHIO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "EFO-RVFL", "class": "OriginalEFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "A-EO-RVFL", "class": "AdaptiveEO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "RIME-RVFL", "class": "OriginalRIME", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "IM-ARO-RVFL", "class": "LARO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "HHO-RVFL", "class": "OriginalHHO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "AIW-PSO-RVFL", "class": "AIW_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "CL-PSO-RVFL", "class": "CL_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    # ]
    #
    # LIST_MODELS = [
    #     {"name": "GA-MLP", "class": "BaseGA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "JADE-MLP", "class": "JADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "L-SHADE-MLP", "class": "L_SHADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "PSO-MLP", "class": "OriginalPSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "HPSO-TVAC-MLP", "class": "HPSO_TVAC", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "P-PSO-MLP", "class": "P_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "RW-GWO-MLP", "class": "RW_GWO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "AO-MLP", "class": "OriginalAO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "ARO-MLP", "class": "OriginalARO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "SMA-MLP", "class": "OriginalSMA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "AOA-MLP", "class": "OriginalAOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "GBO-MLP", "class": "OriginalGBO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "WOA-MLP", "class": "OriginalWOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "ASO-MLP", "class": "OriginalASO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    # ]
