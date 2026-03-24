from models.model_utils import (
    get_ridge_hpo_model,
    run_experiment_suite,
)
from evaluation.generate_test import *
from models.model_utils import (
    get_ridge_hpo_model,
    get_elastic_hpo_model,
    get_rf_hpo_model,
    get_xgb_hpo_model,
    get_lgbm_hpo_model,
    get_catboost_hpo_model,
    get_knn_hpo_model,
    get_dt_hpo_model,
    get_mlp_hpo_model,
)
from feature_engineering.generate_datasets import (
    read_dataset,
    split_train_val,
)

from config import (
    FEATURE_COLS_BASELINE,
    FEATURE_COLS_BASELINE_EXTENDED,
    TARGET_COLS2020,
    TARGET_COLS2021,
    TARGET_COLS_TPLUS1_J,
    TARGET_COLS_DELTA_J,
    FEATURE_COLS_J,
)

"""
TODO:
    first generate a dataset
    call the hpo applied model
"""


def main():
    (
        train_df,
        test_spatial,
    ) = read_dataset("./datasets/change/")

    # only spatial so far
    test_sets = {
        "spatial": test_spatial,
    }

    root_dir = "results_repro"

    model_getters = {
        # "ridge": get_ridge_hpo_model,
        "elastic": get_elastic_hpo_model,
        "rf": get_rf_hpo_model,
        "xgb": get_xgb_hpo_model,
        "lgbm": get_lgbm_hpo_model,
        "catboost": get_catboost_hpo_model,
        "knn": get_knn_hpo_model,
        "dt": get_dt_hpo_model,
        "mlp": get_mlp_hpo_model,
    }

    feature_sets = {
        # "baseline": FEATURE_COLS_BASELINE,
        # "extended": FEATURE_COLS_BASELINE_EXTENDED,
        "extended": FEATURE_COLS_J
    }

    class_names = ["urban", "vegetation", "water", "other"]
    train_df, val_df = split_train_val(train_df)

    print(train_df.columns)
    print("==" * 50)
    print(val_df.columns)

    # model, best_params, study = get_ridge_hpo_model()
    # specify task_type!!
    run_experiment_suite(
        train_df=train_df,
        val_df=val_df,
        test_sets=test_sets,
        feature_sets=feature_sets,
        target_cols=TARGET_COLS_DELTA_J,
        class_names=class_names,
        model_getters=model_getters,
        task_type="change",
        root_dir=root_dir,
    )


if __name__ == "__main__":
    main()
