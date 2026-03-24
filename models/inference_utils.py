import pandas as pd
import pickle
import os

from models.model_utils import normalize_predictions_to_simplex


def load_parquet(file_path: str) -> pd.DataFrame:
    """
    Load a parquet file into a pandas DataFrame.

    Args:
        file_path (str): Path to the parquet file

    Returns:
        pd.DataFrame
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"[LOADED] {file_path} | shape={df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load parquet: {file_path}")
        raise e


def load_model(model_path: str):
    """
    Load a trained model from a .pkl file.

    Args:
        model_path (str): Path to saved model

    Returns:
        model
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print(f"[LOADED MODEL] {model_path}")
        return model

    except Exception as e:
        print(f"[ERROR] Failed to load model: {model_path}")
        raise e


def run_inference_and_format(model, df, feature_cols, task_type):
    TARGET_COLS_TPLUS1_J = [
        "next_built_up",
        "next_vegetation",
        "next_water",
        "next_other",
    ]

    TARGET_COLS_DELTA_J = [
        "delta_built_up",
        "delta_vegetation",
        "delta_water",
        "delta_other",
    ]

    # --- select correct targets ---
    if task_type == "composition":
        target_cols = TARGET_COLS_TPLUS1_J
    elif task_type == "change":
        target_cols = TARGET_COLS_DELTA_J
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # --- predict ---
    X = df[feature_cols]
    y_pred = model.predict(X)

    if task_type == "composition":
        y_pred = normalize_predictions_to_simplex(y_pred)

    # --- base columns (add whatever identifiers you have) ---
    base_cols = []

    if "grid_id" in df.columns:
        base_cols.append("grid_id")

    # optional extras (keep if useful)
    for col in ["image_id", "centroid_x", "centroid_y"]:
        if col in df.columns:
            base_cols.append(col)

    out_df = df[base_cols].copy()

    # --- add predictions ---
    for i, col in enumerate(target_cols):
        out_df[col] = y_pred[:, i]

    return out_df


def save_predictions_parquet(df, save_path):
    """
    Save dataframe as parquet file.

    Args:
        df (pd.DataFrame)
        save_path (str): full path including filename.parquet
    """

    # --- ensure directory exists ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- save ---
    df.to_parquet(save_path, index=False)

    print(f"[SAVED PARQUET] {save_path} | shape={df.shape}")
