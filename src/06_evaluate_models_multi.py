from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from common import ensure_parent, get_logger, load_config, resolve_from_config

CONFIG_PATH = Path("../config/project_config.yaml")
logger = get_logger("06_evaluate_models_dual")
LABEL_BASES = ["built_up", "vegetation", "water", "other"]
STATIC_FEATURES = ["centroid_x", "centroid_y", "area_m2"]


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


def false_change_rate(y_true_delta: np.ndarray, y_pred_delta: np.ndarray, threshold: float) -> float:
    true_change = np.abs(y_true_delta).sum(axis=1) >= threshold
    pred_change = np.abs(y_pred_delta).sum(axis=1) >= threshold
    false_change = np.logical_and(pred_change, ~true_change).sum()
    predicted_change = max(int(pred_change.sum()), 1)
    return float(false_change / predicted_change)


def stability_score(y_true_delta: np.ndarray, y_pred_delta: np.ndarray, threshold: float) -> float:
    true_stable = np.abs(y_true_delta).sum(axis=1) < threshold
    pred_stable = np.abs(y_pred_delta).sum(axis=1) < threshold
    return float((true_stable == pred_stable).mean())


def stress_test_missingness(pipe, X_test: pd.DataFrame, y_test: pd.DataFrame, seed: int, value_name: str) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    base_pred = pipe.predict(X_test)
    rows.append({"stress_type": "missingness", "fraction": 0.0, value_name: float(mean_absolute_error(y_test, base_pred))})
    for frac in [0.05, 0.10, 0.20, 0.30]:
        X_noisy = X_test.copy()
        X_noisy = X_noisy.mask(rng.random(X_noisy.shape) < frac)
        pred = pipe.predict(X_noisy)
        rows.append({"stress_type": "missingness", "fraction": frac, value_name: float(mean_absolute_error(y_test, pred))})
    return pd.DataFrame(rows)


def estimate_rf_uncertainty(bundle: dict, X_test: pd.DataFrame, prefix: str) -> pd.DataFrame:
    pipe = bundle["pipeline"]
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    Xt = preprocessor.transform(X_test)

    out = pd.DataFrame(index=X_test.index)
    for target_name, estimator in zip(bundle["target_columns"], model.estimators_):
        tree_preds = np.vstack([tree.predict(Xt) for tree in estimator.estimators_])
        out[f"{prefix}_uncertainty_std_{target_name}"] = tree_preds.std(axis=0)
        out[f"{prefix}_uncertainty_p10_{target_name}"] = np.percentile(tree_preds, 10, axis=0)
        out[f"{prefix}_uncertainty_p90_{target_name}"] = np.percentile(tree_preds, 90, axis=0)
    return out


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    threshold = float(cfg["modeling"]["change_threshold"])
    anchor_year = int(cfg["modeling"].get("forecast_anchor_year", 2020))
    target_year = int(cfg["modeling"].get("forecast_target_year", anchor_year + 1))

    modeling = pd.read_parquet(resolve_from_config(cfg, cfg["paths"]["modeling_table"]))
    split_manifest = pd.read_parquet(resolve_from_config(cfg, "../artifacts/predictions/split_manifest.parquet"))
    modeling = modeling.merge(split_manifest[["grid_id", "split"]], on="grid_id", how="inner")

    X_all = build_anchor_features(modeling, anchor_year=anchor_year)
    test_df = modeling[modeling["split"] == "test"].copy()
    X_test = X_all.loc[test_df.index]

    y_next = pd.DataFrame(
        {f"next_{base}": test_df[f"{base}_{target_year}"].values for base in LABEL_BASES},
        index=test_df.index,
    )
    y_delta = pd.DataFrame(
        {f"delta_{base}": test_df[f"{base}_{target_year}"].values - test_df[f"{base}_{anchor_year}"].values for base in LABEL_BASES},
        index=test_df.index,
    )
    curr_test = pd.DataFrame(
        {base: test_df[f"{base}_{anchor_year}"].values for base in LABEL_BASES},
        index=test_df.index,
    )

    registry_path = resolve_from_config(cfg, cfg["paths"]["dual_model_registry"])
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    evaluation_dir = resolve_from_config(cfg, "../artifacts/evaluation")
    ensure_parent(evaluation_dir / "x")

    summary_rows = []
    uncertainty_out = pd.DataFrame({"grid_id": test_df["grid_id"].values})

    for key, meta in registry.items():
        bundle = joblib.load(meta["path"])
        task_name = bundle["task_name"]
        model_name = bundle["model_name"]
        pipe = bundle["pipeline"]

        if task_name == "tplus1":
            y_true = y_next
            pred_next = pipe.predict(X_test)
            pred_delta = pred_next - curr_test.to_numpy()
            true_delta = y_delta.to_numpy()

            summary_rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "macro_mae_next_year": float(mean_absolute_error(y_true, pred_next)),
                    "macro_mae_delta": float(mean_absolute_error(true_delta, pred_delta)),
                    "false_change_rate": false_change_rate(true_delta, pred_delta, threshold),
                    "stability_score": stability_score(true_delta, pred_delta, threshold),
                }
            )
            stress = stress_test_missingness(pipe, X_test, y_true, int(cfg["random_seed"]), "macro_mae_next_year")

        else:
            y_true = y_delta
            pred_delta = pipe.predict(X_test)
            pred_next = curr_test.to_numpy() + pred_delta
            true_delta = y_true.to_numpy()

            summary_rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "macro_mae_delta": float(mean_absolute_error(y_true, pred_delta)),
                    "macro_mae_next_year": float(mean_absolute_error(y_next, pred_next)),
                    "false_change_rate": false_change_rate(true_delta, pred_delta, threshold),
                    "stability_score": stability_score(true_delta, pred_delta, threshold),
                }
            )
            stress = stress_test_missingness(pipe, X_test, y_true, int(cfg["random_seed"]), "macro_mae_delta")

        stress.insert(0, "model", model_name)
        stress.insert(0, "task", task_name)
        stress.to_csv(evaluation_dir / f"{task_name}_{model_name}_stress_test.csv", index=False)

        if model_name == "random_forest":
            prefix = "forecast" if task_name == "tplus1" else "direct"
            unc = estimate_rf_uncertainty(bundle, X_test, prefix=prefix)
            uncertainty_out = pd.concat([uncertainty_out.reset_index(drop=True), unc.reset_index(drop=True)], axis=1)

    summary = pd.DataFrame(summary_rows).sort_values(["task", "model"]).reset_index(drop=True)
    summary.to_csv(resolve_from_config(cfg, cfg["paths"]["dual_evaluation_summary"]), index=False)
    uncertainty_out.to_parquet(resolve_from_config(cfg, cfg["paths"]["uncertainty_table"]), index=False)
    logger.info("Saved dual evaluation summary and uncertainty table")


if __name__ == "__main__":
    main()