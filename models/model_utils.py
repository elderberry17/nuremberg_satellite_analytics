import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import json
import pickle
from datetime import datetime


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

from evaluation.metric_utils import (
    evaluate_metrics,
    plot_stress_test,
    save_scatter_pred_vs_true,
)

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


def get_ridge_hpo_model(
    X_train,
    y_train,
    X_val,
    y_val,
    class_names,
    task_type="composition",
    n_trials=20,
    seed=42,
):
    best_result = {"score": float("inf"), "metrics": None, "y_pred": None}

    def objective(trial):
        nonlocal best_result

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

        if task_type == "composition":
            y_pred = normalize_predictions_to_simplex(y_pred)

        metrics = evaluate_metrics(y_val, y_pred, class_names, task_type)

        score = (
            metrics["overall"]["rmse_macro"]
            if task_type == "composition"
            else 1 - metrics["overall"]["direction_accuracy"]
        )

        if score < best_result["score"]:
            best_result.update(
                {"score": score, "metrics": metrics, "y_pred": y_pred}
            )

        return score

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(**study.best_params)),
        ]
    )

    return best_model, study.best_params, study, best_result


def get_rf_hpo_model(
    X_train,
    y_train,
    X_val,
    y_val,
    class_names,
    task_type="composition",
    n_trials=20,
    seed=42,
):
    best_result = {"score": float("inf"), "metrics": None, "y_pred": None}

    def objective(trial):
        nonlocal best_result

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

        if task_type == "composition":
            y_pred = normalize_predictions_to_simplex(y_pred)

        metrics = evaluate_metrics(y_val, y_pred, class_names, task_type)

        score = (
            metrics["overall"]["rmse_macro"]
            if task_type == "composition"
            else 1 - metrics["overall"]["direction_accuracy"]
        )

        if score < best_result["score"]:
            best_result.update(
                {"score": score, "metrics": metrics, "y_pred": y_pred}
            )

        return score

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    **study.best_params, random_state=seed, n_jobs=1
                ),
            ),
        ]
    )

    return best_model, study.best_params, study, best_result


def get_xgb_hpo_model(
    X_train,
    y_train,
    X_val,
    y_val,
    class_names,
    task_type="composition",
    n_trials=20,
    seed=42,
):
    best_result = {
        "score": float("inf"),
        "metrics": None,
        "y_pred": None,
    }

    def objective(trial):
        nonlocal best_result

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
                            **params,
                            random_state=seed,
                            n_jobs=1,
                            verbosity=0,
                        )
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if task_type == "composition":
            y_pred = normalize_predictions_to_simplex(y_pred)

        metrics = evaluate_metrics(y_val, y_pred, class_names, task_type)

        if task_type == "composition":
            score = metrics["overall"]["rmse_macro"]
        elif task_type == "change":
            score = 1 - metrics["overall"]["direction_accuracy"]
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        if score < best_result["score"]:
            best_result["score"] = score
            best_result["metrics"] = metrics
            best_result["y_pred"] = y_pred

        return score

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
                        **best_params,
                        random_state=seed,
                        n_jobs=1,
                        verbosity=0,
                    )
                ),
            ),
        ]
    )

    return best_model, best_params, study, best_result


def get_lgbm_hpo_model(
    X_train,
    y_train,
    X_val,
    y_val,
    class_names,
    task_type="composition",
    n_trials=20,
    seed=42,
):
    best_result = {
        "score": float("inf"),
        "metrics": None,
        "y_pred": None,
    }

    def objective(trial):
        nonlocal best_result

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "min_gain_to_split": trial.suggest_float(
                "min_gain_to_split", 0.0, 1.0
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

        if task_type == "composition":
            y_pred = normalize_predictions_to_simplex(y_pred)

        metrics = evaluate_metrics(y_val, y_pred, class_names, task_type)

        if task_type == "composition":
            score = metrics["overall"]["rmse_macro"]
        elif task_type == "change":
            score = 1 - metrics["overall"]["direction_accuracy"]
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        if score < best_result["score"]:
            best_result["score"] = score
            best_result["metrics"] = metrics
            best_result["y_pred"] = y_pred

        return score

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

    return best_model, best_params, study, best_result


def get_catboost_hpo_model(
    X_train,
    y_train,
    X_val,
    y_val,
    class_names,
    task_type="composition",
    n_trials=20,
    seed=42,
):
    best_result = {
        "score": float("inf"),
        "metrics": None,
        "y_pred": None,
    }

    def objective(trial):
        nonlocal best_result

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

        if task_type == "composition":
            y_pred = normalize_predictions_to_simplex(y_pred)

        metrics = evaluate_metrics(y_val, y_pred, class_names, task_type)

        if task_type == "composition":
            score = metrics["overall"]["rmse_macro"]
        elif task_type == "change":
            score = 1 - metrics["overall"]["direction_accuracy"]
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        if score < best_result["score"]:
            best_result["score"] = score
            best_result["metrics"] = metrics
            best_result["y_pred"] = y_pred

        return score

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

    return best_model, best_params, study, best_result


def get_knn_hpo_model(
    X_train,
    y_train,
    X_val,
    y_val,
    class_names,
    task_type="composition",
    n_trials=20,
    seed=42,
):
    best_result = {
        "score": float("inf"),
        "metrics": None,
        "y_pred": None,
    }

    def objective(trial):
        nonlocal best_result

        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),
            "p": trial.suggest_int("p", 1, 2),
        }

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", KNeighborsRegressor(**params)),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if task_type == "composition":
            y_pred = normalize_predictions_to_simplex(y_pred)

        metrics = evaluate_metrics(y_val, y_pred, class_names, task_type)

        if task_type == "composition":
            score = metrics["overall"]["rmse_macro"]
        elif task_type == "change":
            score = 1 - metrics["overall"]["direction_accuracy"]
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        if score < best_result["score"]:
            best_result["score"] = score
            best_result["metrics"] = metrics
            best_result["y_pred"] = y_pred

        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", KNeighborsRegressor(**best_params)),
        ]
    )

    return best_model, best_params, study, best_result


def get_dt_hpo_model(
    X_train,
    y_train,
    X_val,
    y_val,
    class_names,
    task_type="composition",
    n_trials=20,
    seed=42,
):
    best_result = {
        "score": float("inf"),
        "metrics": None,
        "y_pred": None,
    }

    def objective(trial):
        nonlocal best_result

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

        if task_type == "composition":
            y_pred = normalize_predictions_to_simplex(y_pred)

        metrics = evaluate_metrics(y_val, y_pred, class_names, task_type)

        if task_type == "composition":
            score = metrics["overall"]["rmse_macro"]
        elif task_type == "change":
            score = 1 - metrics["overall"]["direction_accuracy"]
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        if score < best_result["score"]:
            best_result["score"] = score
            best_result["metrics"] = metrics
            best_result["y_pred"] = y_pred

        return score

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

    return best_model, best_params, study, best_result


def get_mlp_hpo_model(
    X_train,
    y_train,
    X_val,
    y_val,
    class_names,
    task_type="composition",
    n_trials=20,
    seed=42,
):
    best_result = {
        "score": float("inf"),
        "metrics": None,
        "y_pred": None,
    }

    layer_map = {
        "64": (64,),
        "128": (128,),
        "64_64": (64, 64),
        "128_64": (128, 64),
    }

    def objective(trial):
        nonlocal best_result

        layer_choice = trial.suggest_categorical(
            "hidden_layer_sizes", ["64", "128", "64_64", "128_64"]
        )

        params = {
            "hidden_layer_sizes": layer_map[layer_choice],
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
        }

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    MLPRegressor(
                        **params,
                        max_iter=300,
                        random_state=seed,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if task_type == "composition":
            y_pred = normalize_predictions_to_simplex(y_pred)

        metrics = evaluate_metrics(y_val, y_pred, class_names, task_type)

        if task_type == "composition":
            score = metrics["overall"]["rmse_macro"]
        elif task_type == "change":
            score = 1 - metrics["overall"]["direction_accuracy"]
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        if score < best_result["score"]:
            best_result["score"] = score
            best_result["metrics"] = metrics
            best_result["y_pred"] = y_pred

        return score

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    best_params["hidden_layer_sizes"] = layer_map[
        best_params["hidden_layer_sizes"]
    ]

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "regressor",
                MLPRegressor(
                    **best_params,
                    max_iter=300,
                    random_state=seed,
                ),
            ),
        ]
    )

    return best_model, best_params, study, best_result


def fit_and_predict(
    model,
    train_df,
    test_df,
    feature_cols,
    target_cols,
    task_type="composition",
):
    rng = np.random.default_rng(SEED)

    # prepare data
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols].to_numpy()

    X_test = test_df[feature_cols]
    y_test = test_df[target_cols].to_numpy()

    # train
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # predict_noisy
    noise_light = rng.normal(
        loc=0.0,
        scale=0.01 * X_test.std().values,
        size=X_test.shape,
    )

    noise_strong = rng.normal(
        loc=0.0,
        scale=0.1 * X_test.std().values,
        size=X_test.shape,
    )

    y_pred_noise_light = model.predict(X_test + noise_light)
    y_pred_noise_strong = model.predict(X_test + noise_strong)

    # normalize to distribution
    if task_type == "composition":
        y_pred = normalize_predictions_to_simplex(y_pred)
        y_pred_noise_light = normalize_predictions_to_simplex(
            y_pred_noise_light
        )
        y_pred_noise_strong = normalize_predictions_to_simplex(
            y_pred_noise_strong
        )

    return model, y_test, y_pred, y_pred_noise_light, y_pred_noise_strong


def run_experiment_suite(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_sets: dict[str, pd.DataFrame],
    feature_sets: dict[str, list[str]],
    target_cols: list[str],
    class_names: list[str],
    model_getters: dict[str, callable],
    root_dir: str,
    task_type="composition",
):
    all_results = []
    experiment_id = datetime.now().strftime(
        f"exp_%Y-%m-%d_%H-%M-%S-{task_type}"
    )

    for model_name, get_model_fn in model_getters.items():
        print("=" * 60)
        print(f"MODEL: {model_name}")

        for feature_set_name, feature_cols in feature_sets.items():
            print("-" * 60)
            print(
                f"FEATURE SET: {feature_set_name} ({len(feature_cols)} feat.s)"
            )

            # prepare train arrays
            X_train = train_df[feature_cols]
            y_train = train_df[target_cols].to_numpy()

            X_val = val_df[feature_cols]
            y_val = val_df[target_cols].to_numpy()

            print(f"HPO for {model_name}")
            # HPO (returns best model)
            model, best_params, _, val_result = get_model_fn(
                X_train, y_train, X_val, y_val, class_names, task_type
            )

            print(f"Best params: {best_params}")
            assert (
                val_result["y_pred"] is not None
            ), "Validation predictions missing"

            val_preds_df = save_predictions(
                test_df=val_df,
                y_true=y_val,
                y_pred=val_result["y_pred"],
                target_cols=target_cols,
            )

            for test_name, test_df in test_sets.items():
                print(f"Training with test set: {test_name}")

                # fit + predict
                (
                    model,
                    y_true,
                    y_pred,
                    y_pred_noise_light,
                    y_pred_noise_strong,
                ) = fit_and_predict(
                    model,
                    train_df,
                    test_df,
                    feature_cols,
                    target_cols,
                    task_type,
                )

                # evaluate
                result = evaluate_metrics(
                    y_true, y_pred, class_names, task_type
                )

                stress_result_light = evaluate_metrics(
                    y_true, y_pred_noise_light, class_names, task_type
                )

                stress_result_strong = evaluate_metrics(
                    y_true, y_pred_noise_strong, class_names, task_type
                )

                preds_df = save_predictions(
                    test_df=test_df,
                    y_true=y_true,
                    y_pred=y_pred,
                    target_cols=target_cols,
                )

                log_experiment(
                    model_name=model_name,
                    task_type=task_type,
                    target_cols=target_cols,
                    model=model,
                    feature_set_name=feature_set_name,
                    test_name=test_name,
                    params=best_params,
                    result=result,
                    stress_result_light=stress_result_light,
                    stress_result_strong=stress_result_strong,
                    preds_df=preds_df,
                    val_result=val_result["metrics"],
                    val_preds_df=val_preds_df,
                    experiment_id=experiment_id,
                    root_dir=root_dir,
                    metadata={
                        "n_features": len(feature_cols),
                        "train_size": len(train_df),
                        "val_size": len(val_df),
                        "test_size": len(test_df),
                    },
                )

                print(f"\nResults for test set: {test_name}")
                print(result["overall"])

                row = {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "test_set": test_name,
                    "n_features": len(feature_cols),
                }

                row.update(result["overall"])

                all_results.append(row)

    return pd.DataFrame(all_results)


def log_experiment(
    model_name,
    task_type,
    target_cols,
    model,
    feature_set_name,
    test_name,
    params,
    result,
    stress_result_light,
    stress_result_strong,
    preds_df,
    val_result,
    val_preds_df,
    experiment_id,
    root_dir="results",
    metadata=None,
):
    # --- timestamp ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- directory: results/model_name/ ---
    exp_dir = os.path.join(root_dir, experiment_id)
    model_dir = os.path.join(exp_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # --- filename ---
    filename = f"{task_type}_{model_name}_{feature_set_name}_{test_name}_{timestamp}.json"
    filepath = os.path.join(model_dir, filename)

    # --- build log dict ---
    # need to save weights
    log_data = {
        "model": model_name,
        "feature_set": feature_set_name,
        "test_set": test_name,
        "timestamp": timestamp,
        "params": params,
        "result": result,
        "stress_result_light": stress_result_light,
        "stress_result_strong": stress_result_strong,
        "metadata": metadata or {},
        "task_type": task_type,
    }

    model_filename = filename.replace(".json", ".pkl")
    model_path = os.path.join(model_dir, model_filename)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    log_data["model_path"] = model_path

    log_data["val_result"] = val_result
    val_preds_filename = filename.replace(".json", "_val_preds.csv")
    val_preds_path = os.path.join(model_dir, val_preds_filename)

    val_preds_df.to_csv(val_preds_path, index=False)
    log_data["val_predictions_path"] = val_preds_path

    preds_filename = filename.replace(".json", "_preds.csv")
    preds_path = os.path.join(model_dir, preds_filename)
    preds_df.to_csv(preds_path, index=False)
    log_data["predictions_path"] = preds_path

    stress_plot_filename = filename.replace(".json", "_stress.png")
    stress_plot_path = os.path.join(model_dir, stress_plot_filename)

    plot_stress_test(
        result=result,
        stress_result_light=stress_result_light,
        stress_result_strong=stress_result_strong,
        model_name=model_name,
        feature_set_name=feature_set_name,
        test_name=test_name,
        task_type=task_type,
        save_path=stress_plot_path,
    )

    scatter_plot_filename = filename.replace(".json", "_scatter.png")
    scatter_plot_path = os.path.join(model_dir, scatter_plot_filename)

    save_scatter_pred_vs_true(
        y_true=preds_df[[f"true_{c}" for c in target_cols]].values,
        y_pred=preds_df[[f"pred_{c}" for c in target_cols]].values,
        class_names=target_cols,
        model_name=model_name,
        save_path=scatter_plot_path,
    )

    log_data["scatter_plot"] = scatter_plot_path

    # --- save ---
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"[LOGGED] {filepath}")


def save_predictions(test_df, y_true, y_pred, target_cols):
    assert y_pred.shape == y_true.shape

    df = test_df[["centroid_x", "centroid_y"]].copy()
    pred_cols = [f"pred_{col}" for col in target_cols]
    true_cols = [f"true_{col}" for col in target_cols]

    for i, col in enumerate(pred_cols):
        df[col] = y_pred[:, i]

    # Add ground truth
    for i, col in enumerate(true_cols):
        df[col] = y_true[:, i]

    return df
