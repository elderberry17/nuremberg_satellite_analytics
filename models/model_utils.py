import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

import optuna
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from evaluation.metric_utils import evaluate_all_metrics

from config import ROOT_NAME

SEED = 42
ROOT = Path(os.path.join("../process_esa/", ROOT_NAME))

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def get_params():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def normalize_predictions_to_simplex(
    y_pred: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """
    Clip predictions to [0, +inf) and renormalize rows to sum to 1.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, 0.0, None)

    row_sums = y_pred.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(-1) < eps

    # fallback: uniform distribution if model predicted all zeros
    if np.any(zero_rows):
        y_pred[zero_rows] = 1.0
        row_sums = y_pred.sum(axis=1, keepdims=True)

    y_pred = y_pred / row_sums
    return y_pred


def get_ridge_hpo_model(X_train, y_train, X_val, y_val, n_trials=20, seed=42):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 100, log=True)

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=alpha)),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(**best_params)),
        ]
    )

    return best_model, best_params, study


def get_rf_hpo_model(X_train, y_train, X_val, y_val, n_trials=20, seed=42):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        }

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    RandomForestRegressor(
                        **params, random_state=seed, n_jobs=1
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    **best_params, random_state=seed, n_jobs=1
                ),
            ),
        ]
    )

    return best_model, best_params, study


def get_xgb_hpo_model(X_train, y_train, X_val, y_val, n_trials=20, seed=42):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
        }

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    MultiOutputRegressor(
                        XGBRegressor(
                            **params, random_state=seed, n_jobs=1, verbosity=0
                        )
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                MultiOutputRegressor(
                    XGBRegressor(
                        **best_params, random_state=seed, n_jobs=1, verbosity=0
                    )
                ),
            ),
        ]
    )

    return best_model, best_params, study


def get_lgbm_hpo_model(X_train, y_train, X_val, y_val, n_trials=20, seed=42):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
        }

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    MultiOutputRegressor(
                        LGBMRegressor(**params, random_state=seed, n_jobs=1)
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                MultiOutputRegressor(
                    LGBMRegressor(**best_params, random_state=seed, n_jobs=1)
                ),
            ),
        ]
    )

    return best_model, best_params, study


def get_catboost_hpo_model(
    X_train, y_train, X_val, y_val, n_trials=20, seed=42
):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 400),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        }

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    MultiOutputRegressor(
                        CatBoostRegressor(
                            **params, random_seed=seed, verbose=0
                        )
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                MultiOutputRegressor(
                    CatBoostRegressor(
                        **best_params, random_seed=seed, verbose=0
                    )
                ),
            ),
        ]
    )

    return best_model, best_params, study


def get_knn_hpo_model(X_train, y_train, X_val, y_val, n_trials=20, seed=42):
    def objective(trial):
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),
            "p": trial.suggest_int("p", 1, 2),  # 1=manhattan, 2=euclidean
        }

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),  # IMPORTANT for KNN
                ("regressor", KNeighborsRegressor(**params)),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", KNeighborsRegressor(**best_params)),
        ]
    )

    return best_model, best_params, study


def get_dt_hpo_model(X_train, y_train, X_val, y_val, n_trials=20, seed=42):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        }

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    DecisionTreeRegressor(**params, random_state=seed),
                ),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                DecisionTreeRegressor(**best_params, random_state=seed),
            ),
        ]
    )

    return best_model, best_params, study


def get_mlp_hpo_model(X_train, y_train, X_val, y_val, n_trials=20, seed=42):
    def objective(trial):
        params = {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", [(64,), (128,), (64, 64), (128, 64)]
            ),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
        }

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),  # VERY IMPORTANT for MLP
                (
                    "regressor",
                    MLPRegressor(**params, max_iter=300, random_state=seed),
                ),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "regressor",
                MLPRegressor(**best_params, max_iter=300, random_state=seed),
            ),
        ]
    )

    return best_model, best_params, study


def fit_and_predict(
    model,
    train_df,
    test_df,
    feature_cols,
    target_cols,
):
    # prepare data
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols].to_numpy()

    X_test = test_df[feature_cols]
    y_test = test_df[target_cols].to_numpy()

    # train
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # normalize to distribution
    y_pred = normalize_predictions_to_simplex(y_pred)

    return model, y_test, y_pred


def run_experiment_suite(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_sets: dict[str, pd.DataFrame],
    feature_sets: dict[str, list[str]],
    target_cols: list[str],
    class_names: list[str],
    model_getters: dict[str, callable],
):
    all_results = []

    for model_name, get_model_fn in model_getters.items():
        print("=" * 120)
        print(f"MODEL: {model_name}")

        for feature_set_name, feature_cols in feature_sets.items():
            print("-" * 120)
            print(
                f"FEATURE SET: {feature_set_name} ({len(feature_cols)} feat.s)"
            )

            # prepare train arrays
            X_train = train_df[feature_cols]
            y_train = train_df[target_cols].to_numpy()

            X_val = val_df[feature_cols]
            y_val = val_df[target_cols].to_numpy()

            # HPO (returns best model)
            model, best_params, _ = get_model_fn(
                X_train, y_train, X_val, y_val
            )

            print(f"Best params: {best_params}")

            for test_name, test_df in test_sets.items():
                # fit + predict
                _, y_true, y_pred = fit_and_predict(
                    model,
                    train_df,
                    test_df,
                    feature_cols,
                    target_cols,
                )

                # evaluate
                result = evaluate_all_metrics(y_true, y_pred, class_names)

                print(f"\nTest set: {test_name}")
                print(result["overall"])

                all_results.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "test_set": test_name,
                        "n_features": len(feature_cols),
                        "mae_macro": result["overall"]["mae_macro"],
                        "rmse_macro": result["overall"]["rmse_macro"],
                        "r2_macro": result["overall"]["r2_macro"],
                        "kl": result["overall"]["kl"],
                    }
                )

    return pd.DataFrame(all_results)
