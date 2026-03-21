from models.model_utils import (
    get_ridge_hpo_model,
    run_experiment_suite,
)
from evaluation.generate_test import *
from models.model_utils import (
    get_ridge_hpo_model,
    get_rf_hpo_model,
    get_xgb_hpo_model,
    get_lgbm_hpo_model,
    get_catboost_hpo_model,
    get_knn_hpo_model,
    get_dt_hpo_model,
    get_mlp_hpo_model,
)
from feature_engineering.generate_datasets import read_train_test_datasets, split_train_val

from config import (
    FEATURE_COLS_BASELINE,
    FEATURE_COLS_BASELINE_EXTENDED,
    TARGET_COLS2020,
    TARGET_COLS2021,
)

"""
TODO:
    first generate a dataset
    call the hpo applied model
"""


def main():
    train_df, test_spatial = read_train_test_datasets(
        "data/v1/train_df.pq",
        "data/v1/test_spatial.pq"
    )

    # only spatial so far
    test_sets = {
        "spatial": test_spatial,
        # "spatial_temporal": test_spatial_temporal,
    }

    model_getters = {
        "ridge": get_ridge_hpo_model,
        "rf": get_rf_hpo_model,
        "xgb": get_xgb_hpo_model,
        "lgbm": get_lgbm_hpo_model,
        "catboost": get_catboost_hpo_model,
        "knn": get_knn_hpo_model,
        "dt": get_dt_hpo_model,
        "mlp": get_mlp_hpo_model,
    }

    feature_sets = {
        "baseline": FEATURE_COLS_BASELINE,
        "extended": FEATURE_COLS_BASELINE_EXTENDED,
    }
    class_names = ["urban", "water", "vegetation"]
    train_df, val_df = split_train_val(train_df)

    # model, best_params, study = get_ridge_hpo_model()
    run_experiment_suite(
        train_df=train_df,
        val_df=val_df,
        test_sets=test_sets,
        feature_sets=feature_sets,
        target_cols=TARGET_COLS2021,
        class_names=class_names,
        model_getters=model_getters,
    )


if __name__ == "__main__":
    main()
