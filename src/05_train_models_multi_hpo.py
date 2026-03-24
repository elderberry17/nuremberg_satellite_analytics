from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import dump_json, ensure_parent, get_logger, load_config, resolve_from_config

CONFIG_PATH = Path("../config/project_config.yaml")
logger = get_logger("05_train_models_dual")
LABEL_BASES = ["built_up", "vegetation", "water", "other"]
STATIC_FEATURES = ["centroid_x", "centroid_y", "area_m2"]


def assign_spatial_blocks(df: pd.DataFrame, block_size_m: float) -> pd.DataFrame:
    out = df.copy()
    out["block_x"] = np.floor(out["centroid_x"] / block_size_m).astype(int)
    out["block_y"] = np.floor(out["centroid_y"] / block_size_m).astype(int)
    out["spatial_block"] = out["block_x"].astype(str) + "_" + out["block_y"].astype(str)
    return out


def make_split(df: pd.DataFrame, test_fraction: float, seed: int) -> pd.DataFrame:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    _, test_idx = next(gss.split(df, groups=df["spatial_block"]))
    out = df.copy()
    out["split"] = "train"
    out.loc[df.index[test_idx], "split"] = "test"
    return out


def make_inner_split(df: pd.DataFrame, validation_fraction: float, seed: int) -> pd.DataFrame:
    gss = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df["spatial_block"]))
    out = df.copy()
    out["inner_split"] = "train"
    out.loc[df.index[val_idx], "inner_split"] = "val"
    return out


def build_anchor_features(df: pd.DataFrame, anchor_year: int) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in STATIC_FEATURES:
        if col in df.columns:
            X[col] = df[col]

    suffix = f"_{anchor_year}"
    for col in df.columns:
        if not col.endswith(suffix):
            continue
        base = col[: -len(suffix)]
        if base in LABEL_BASES:
            X[f"curr_label__{base}"] = df[col]
        elif not base.startswith("delta_"):
            X[base] = df[col]
    return X


def build_targets(df: pd.DataFrame, anchor_year: int, target_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    next_year = pd.DataFrame(index=df.index)
    deltas = pd.DataFrame(index=df.index)
    for base in LABEL_BASES:
        next_col = f"{base}_{target_year}"
        curr_col = f"{base}_{anchor_year}"
        if next_col not in df.columns or curr_col not in df.columns:
            raise ValueError(f"Missing required columns for {base}: {curr_col}, {next_col}")
        next_year[f"next_{base}"] = df[next_col]
        deltas[f"delta_{base}"] = df[next_col] - df[curr_col]
    return next_year, deltas


def build_pipelines(
    feature_cols: list[str],
    cfg: dict,
    overrides: dict[str, dict] | None = None,
) -> dict[str, Pipeline]:
    overrides = overrides or {}

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ]
    )

    e_cfg = {**cfg["training"]["elastic_net"], **overrides.get("elastic_net", {})}
    rf_cfg = {**cfg["training"]["random_forest"], **overrides.get("random_forest", {})}
    gb_cfg = {**cfg["training"].get("gradient_boosting", {}), **overrides.get("gradient_boosting", {})}

    elastic = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                MultiOutputRegressor(
                    ElasticNet(
                        alpha=float(e_cfg["alpha"]),
                        l1_ratio=float(e_cfg["l1_ratio"]),
                        max_iter=int(e_cfg["max_iter"]),
                        random_state=int(cfg["random_seed"]),
                    )
                ),
            ),
        ]
    )

    rf = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                MultiOutputRegressor(
                    RandomForestRegressor(
                        n_estimators=int(rf_cfg["n_estimators"]),
                        max_depth=None if rf_cfg.get("max_depth", None) is None else int(rf_cfg["max_depth"]),
                        min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
                        n_jobs=int(rf_cfg["n_jobs"]),
                        random_state=int(cfg["random_seed"]),
                    )
                ),
            ),
        ]
    )

    gb = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        learning_rate=float(gb_cfg.get("learning_rate", 0.05)),
                        max_iter=int(gb_cfg.get("max_iter", 300)),
                        max_depth=None if gb_cfg.get("max_depth", None) is None else int(gb_cfg["max_depth"]),
                        min_samples_leaf=int(gb_cfg.get("min_samples_leaf", 20)),
                        l2_regularization=float(gb_cfg.get("l2_regularization", 0.0)),
                        random_state=int(cfg["random_seed"]),
                    )
                ),
            ),
        ]
    )

    return {
        "elastic_net": elastic,
        "random_forest": rf,
        "gradient_boosting": gb,
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(target_names):
        rows.append(
            {
                "target": t,
                "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
                "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
                "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
            }
        )
    rows.append(
        {
            "target": "macro_avg",
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        }
    )
    return pd.DataFrame(rows)


def optimize_hyperparameters(
    model_name: str,
    feature_cols: list[str],
    cfg: dict,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    task_name: str,
) -> tuple[dict, float]:
    hpo_cfg = cfg["training"].get("hpo", {})
    n_trials = int(hpo_cfg.get("n_trials", 20))
    optimize_metric = hpo_cfg.get("optimize_metric", "mae")

    def objective(trial: optuna.Trial) -> float:
        if model_name == "elastic_net":
            overrides = {
                "elastic_net": {
                    "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
                    "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
                    "max_iter": 20000,
                }
            }
        elif model_name == "random_forest":
            overrides = {
                "random_forest": {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
                    "max_depth": trial.suggest_int("max_depth", 6, 24),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
                    "n_jobs": cfg["training"]["random_forest"]["n_jobs"],
                }
            }
        elif model_name == "gradient_boosting":
            overrides = {
                "gradient_boosting": {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    "max_iter": trial.suggest_int("max_iter", 150, 500, step=50),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 40, step=5),
                    "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1.0, log=True),
                }
            }
        else:
            raise ValueError(f"Unsupported model for HPO: {model_name}")

        pipe = build_pipelines(feature_cols, cfg, overrides=overrides)[model_name]
        pipe.fit(X_train, y_train)
        pred_val = pipe.predict(X_val)

        metric_df = evaluate_predictions(y_val.to_numpy(), pred_val, list(y_val.columns))
        macro_row = metric_df.loc[metric_df["target"] == "macro_avg"].iloc[0]

        if optimize_metric == "rmse":
            return float(macro_row["rmse"])
        return float(macro_row["mae"])

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(cfg["random_seed"])),
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info(
        "Best HPO result for %s / %s -> score=%.6f | params=%s",
        task_name,
        model_name,
        float(study.best_value),
        study.best_params,
    )
    return study.best_params, float(study.best_value)


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    modeling = pd.read_parquet(resolve_from_config(cfg, cfg["paths"]["modeling_table"]))
    anchor_year = int(cfg["modeling"].get("forecast_anchor_year", 2020))
    target_year = int(cfg["modeling"].get("forecast_target_year", anchor_year + 1))

    modeling = assign_spatial_blocks(modeling, float(cfg["modeling"]["spatial_block_size_m"]))
    modeling = make_split(modeling, float(cfg["modeling"]["test_fraction"]), int(cfg["random_seed"]))

    X = build_anchor_features(modeling, anchor_year=anchor_year)
    y_next, y_delta = build_targets(modeling, anchor_year=anchor_year, target_year=target_year)

    task_map = {
        "delta": y_delta,
        "tplus1": y_next,
    }

    artifacts_root = resolve_from_config(cfg, "../artifacts")
    metrics_dir = artifacts_root / "metrics"
    models_dir = artifacts_root / "models"
    preds_dir = artifacts_root / "predictions"
    for p in [metrics_dir / "x", models_dir / "x", preds_dir / "x"]:
        ensure_parent(p)

    split_cols = ["grid_id", "centroid_x", "centroid_y", "spatial_block", "split"]
    modeling[split_cols].to_parquet(preds_dir / "split_manifest.parquet", index=False)

    train_mask = modeling["split"] == "train"
    test_mask = modeling["split"] == "test"

    train_df = modeling.loc[train_mask].copy()
    test_df = modeling.loc[test_mask].copy()

    X_test = X.loc[test_mask]
    current_test = pd.DataFrame(
        {f"curr_{base}": modeling.loc[test_mask, f"{base}_{anchor_year}"].values for base in LABEL_BASES}
    )

    hpo_cfg = cfg["training"].get("hpo", {})
    hpo_enabled = bool(hpo_cfg.get("enabled", False))
    val_fraction = float(hpo_cfg.get("validation_fraction", 0.20))

    registry: dict[str, dict] = {}

    for task_name, y in task_map.items():
        y_test = y.loc[test_mask]

        for model_name in ["elastic_net", "random_forest", "gradient_boosting"]:
            logger.info("Training %s / %s", task_name, model_name)

            overrides: dict[str, dict] = {}
            best_hpo_score = None

            if hpo_enabled:
                inner_train_df = make_inner_split(
                    train_df,
                    validation_fraction=val_fraction,
                    seed=int(cfg["random_seed"]),
                )

                inner_train_mask = inner_train_df["inner_split"] == "train"
                inner_val_mask = inner_train_df["inner_split"] == "val"

                X_inner = build_anchor_features(inner_train_df, anchor_year=anchor_year)
                y_inner_full = y.loc[inner_train_df.index]

                X_inner_train = X_inner.loc[inner_train_mask]
                X_inner_val = X_inner.loc[inner_val_mask]
                y_inner_train = y_inner_full.loc[inner_train_mask]
                y_inner_val = y_inner_full.loc[inner_val_mask]

                best_params, best_hpo_score = optimize_hyperparameters(
                    model_name=model_name,
                    feature_cols=list(X_inner.columns),
                    cfg=cfg,
                    X_train=X_inner_train,
                    y_train=y_inner_train,
                    X_val=X_inner_val,
                    y_val=y_inner_val,
                    task_name=task_name,
                )
                overrides = {model_name: best_params}

            X_train_full = build_anchor_features(train_df, anchor_year=anchor_year)
            y_train_full = y.loc[train_mask]

            pipe = build_pipelines(
                feature_cols=list(X_train_full.columns),
                cfg=cfg,
                overrides=overrides,
            )[model_name]

            pipe.fit(X_train_full, y_train_full)
            pred_test = pipe.predict(X_test)

            metric_df = evaluate_predictions(y_test.to_numpy(), pred_test, list(y.columns))
            metric_df.insert(0, "model", model_name)
            metric_df.insert(0, "task", task_name)
            metric_df.to_csv(metrics_dir / f"{task_name}_{model_name}_metrics.csv", index=False)

            preds = modeling.loc[test_mask, ["grid_id", "centroid_x", "centroid_y"]].copy().reset_index(drop=True)
            preds = pd.concat([preds, current_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

            for i, col in enumerate(y.columns):
                preds[f"pred_{col}"] = pred_test[:, i]
                preds[f"resid_{col}"] = preds[col] - preds[f"pred_{col}"]
                base = col.replace("next_", "").replace("delta_", "")
                if task_name == "tplus1":
                    preds[f"pred_delta_{base}"] = preds[f"pred_{col}"] - preds[f"curr_{base}"]
                    preds[f"true_delta_{base}"] = preds[col] - preds[f"curr_{base}"]
                else:
                    preds[f"pred_next_{base}"] = preds[f"curr_{base}"] + preds[f"pred_{col}"]
                    preds[f"true_next_{base}"] = preds[f"curr_{base}"] + preds[col]

            preds.to_parquet(preds_dir / f"{task_name}_{model_name}_test_predictions.parquet", index=False)

            model_path = models_dir / f"{task_name}_{model_name}.joblib"
            bundle = {
                "pipeline": pipe,
                "feature_columns": list(X_train_full.columns),
                "target_columns": list(y.columns),
                "target_bases": LABEL_BASES,
                "task_name": task_name,
                "anchor_year": anchor_year,
                "target_year": target_year,
                "model_name": model_name,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "hpo_enabled": hpo_enabled,
                "best_hpo_score": best_hpo_score,
                "best_hpo_params": overrides.get(model_name, {}),
            }
            joblib.dump(bundle, model_path)

            registry[f"{task_name}_{model_name}"] = {
                "path": str(model_path),
                "task_name": task_name,
                "model_name": model_name,
                "anchor_year": anchor_year,
                "target_year": target_year,
                "hpo_enabled": hpo_enabled,
                "best_hpo_score": best_hpo_score,
                "best_hpo_params": overrides.get(model_name, {}),
            }

            logger.info("Finished %s / %s", task_name, model_name)

    dump_json(resolve_from_config(cfg, cfg["paths"]["dual_model_registry"]), registry)
    logger.info("Dual training complete. Train=%d | Test=%d", int(train_mask.sum()), int(test_mask.sum()))


if __name__ == "__main__":
    main()