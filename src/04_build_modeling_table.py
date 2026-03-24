from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from common import ensure_parent, get_logger, load_config

CONFIG_PATH = Path('../config/project_config.yaml')
logger = get_logger('04_build_modeling_table')
LABEL_BASES = ['built_up', 'vegetation', 'water', 'other']


def add_coordinate_features(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    grid = grid.copy()
    if 'area_m2' not in grid.columns:
        grid['area_m2'] = grid.geometry.area
    centroids = grid.geometry.centroid
    return pd.DataFrame({
        'grid_id': grid['grid_id'].values,
        'centroid_x': centroids.x.values,
        'centroid_y': centroids.y.values,
        'area_m2': grid['area_m2'].values,
    })


def add_temporal_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols_2020 = [c for c in df.columns if c.endswith('_2020') and c[:-5] + '_2021' in df.columns]
    excluded_prefixes = tuple(LABEL_BASES)
    for c20 in feature_cols_2020:
        base = c20[:-5]
        if base.startswith(excluded_prefixes):
            continue
        df[f'delta_feat__{base}'] = df[f'{base}_2021'] - df[c20]
    return df


def add_change_helpers(df: pd.DataFrame, change_threshold: float) -> pd.DataFrame:
    out = df.copy()
    delta_cols = [f'delta_{base}' for base in LABEL_BASES if f'delta_{base}' in out.columns]
    out['abs_total_change'] = out[delta_cols].abs().sum(axis=1)
    out['change_flag'] = (out['abs_total_change'] >= change_threshold).astype(int)
    out['dominant_change_class'] = out[delta_cols].abs().idxmax(axis=1).str.replace('delta_', '', regex=False)
    return out


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    grid = gpd.read_file(cfg['paths']['grid_file'])
    labels = pd.read_parquet(cfg['paths']['labels_table'])
    features = pd.read_parquet(cfg['paths']['features_table'])
    coords = add_coordinate_features(grid)
    modeling = coords.merge(labels, on='grid_id', how='inner').merge(features, on='grid_id', how='inner')
    modeling = add_temporal_delta_features(modeling)
    modeling = add_change_helpers(modeling, float(cfg['modeling']['change_threshold']))
    ensure_parent(cfg['paths']['modeling_table'])
    modeling.to_parquet(cfg['paths']['modeling_table'], index=False)
    logger.info('Saved modeling table to %s with shape %s', cfg['paths']['modeling_table'], modeling.shape)


if __name__ == '__main__':
    main()
