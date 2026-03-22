import os
from pathlib import Path
import pandas as pd
from scipy.sparse import data
from evaluation.generate_test import spatial_train_test_split
from sklearn.model_selection import train_test_split

from feature_engineering.build_tables import (
    build_split_table_tif,
    collect_dataset_items_tif,
)

from feature_engineering.build_tables_extended import (
    build_split_table_tif_extended,
    collect_items_extended,
)

from config import (
    ROOT_NAME,
    TRAIN_FILES,
    LABEL_PATHS,
    ROOT_NAME,
)

ROOT = Path(os.path.join("../process_esa/", ROOT_NAME))


def generate_default_baseline(
    kernel_size=11, stride=200, keep_borders=True, distribution=True
):
    # train_items, test_items = collect_dataset_items(ROOT, DIRS2USE)
    items = collect_dataset_items_tif(
        root=ROOT, folder_names=TRAIN_FILES, label_path=LABEL_PATHS
    )

    df_baseline = build_split_table_tif(
        items=items,
        kernel_size=kernel_size,
        keep_borders=keep_borders,
        stride=stride,
        distribution=distribution,
    )

    return df_baseline


def generate_extended_baseline(
    kernel_size=11, stride=200, keep_borders=True, distribution=True
):
    # train_items, test_items = collect_dataset_items(ROOT, DIRS2USE)
    items = collect_items_extended(
        root=ROOT, folder_names=TRAIN_FILES, label_path=LABEL_PATHS
    )

    df_extended = build_split_table_tif_extended(
        items=items,
        kernel_size=kernel_size,
        keep_borders=keep_borders,
        stride=stride,
        distribution=distribution,
    )

    return df_extended


def generate_holdout_sets(df_extended, test_temporal):
    train_coors_df, test_coors_df = spatial_train_test_split(
        df_extended, "x", "y"
    )
    train_df = df_extended.merge(train_coors_df, on=["x", "y"], how="inner")

    # same dates, diff xy
    test_spatial = df_extended.merge(test_coors_df, on=["x", "y"], how="inner")

    # same xy, diff dates
    test_temporal_only = test_temporal.merge(
        train_coors_df, on=["x", "y"], how="inner"
    )

    # diff xy, diff dates
    test_spatial_temporal = test_temporal.merge(
        test_coors_df, on=["x", "y"], how="inner"
    )

    return train_df, test_spatial, test_temporal_only, test_spatial_temporal


def read_dataset_deprecated(dataset_dir):
    train_df = pd.read_parquet(dataset_dir)
    test_spatial = pd.read_parquet(dataset_dir)
    test_temporal = pd.read_parquet(dataset_dir)
    test_spatial_temporal = pd.read_parquet(dataset_dir)

    return train_df, test_spatial


def read_dataset(dataset_dir):
    train_path = dataset_dir + "train_df.pq"
    test_spatial_path = dataset_dir + "test_spatial.pq"

    train_df = pd.read_parquet(train_path)
    test_spatial = pd.read_parquet(test_spatial_path)

    return train_df, test_spatial


def split_train_val(
    train_df,
    val_size=0.2,
    seed=42,
    stratify_col=None,
):
    if stratify_col is not None:
        stratify_vals = train_df[stratify_col]
    else:
        stratify_vals = None

    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_vals,
    )

    return train_split.reset_index(drop=True), val_split.reset_index(drop=True)
