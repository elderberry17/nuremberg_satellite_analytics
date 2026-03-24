from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd

from common import ensure_parent, get_logger, load_config, resolve_from_config

CONFIG_PATH = Path('../config/project_config.yaml') # Path(__file__).resolve().with_name('project_config.yaml')
logger = get_logger('07_generate_app_predictions_dual')
LABEL_BASES = ['built_up', 'vegetation', 'water', 'other']
STATIC_FEATURES = ['centroid_x', 'centroid_y', 'area_m2']


def build_anchor_features(df: pd.DataFrame, anchor_year: int) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in STATIC_FEATURES:
        if col in df.columns:
            X[col] = df[col]
    suffix = f'_{anchor_year}'
    for col in df.columns:
        if not col.endswith(suffix):
            continue
        base = col[:-len(suffix)]
        if base in LABEL_BASES:
            X[f'curr_label__{base}'] = df[col]
        elif not base.startswith('delta_'):
            X[base] = df[col]
    return X


def classify_dominant_change(df: pd.DataFrame, prefix: str) -> pd.Series:
    change_cols = [f'{prefix}_delta_{base}' for base in LABEL_BASES if f'{prefix}_delta_{base}' in df.columns]
    if not change_cols:
        return pd.Series(['unknown'] * len(df), index=df.index)
    return df[change_cols].abs().idxmax(axis=1).str.replace(f'{prefix}_delta_', '', regex=False)


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    registry_path = resolve_from_config(cfg, cfg['paths']['dual_model_registry'])
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    grid = gpd.read_file(resolve_from_config(cfg, cfg['paths']['grid_file']))
    modeling = pd.read_parquet(resolve_from_config(cfg, cfg['paths']['modeling_table']))

    observed_years = list(cfg['modeling'].get('app_observed_years', [2020, 2021]))
    inference_anchor_year = int(cfg['modeling'].get('app_inference_anchor_year', max(observed_years)))
    forecast_year = int(cfg['modeling'].get('app_forecast_year', inference_anchor_year + 1))
    final_model_name = cfg['modeling'].get('app_final_model_name', 'random_forest')

    final_tplus1_model_name = cfg['modeling'].get('app_final_model_name_tplus1', 'random_forest')
    final_delta_model_name = cfg['modeling'].get('app_final_model_name_delta', 'gradient_boosting')

    out = pd.DataFrame({'grid_id': modeling['grid_id'].values})
    for year in observed_years:
        for base in LABEL_BASES:
            col = f'{base}_{year}'
            if col in modeling.columns:
                out[f'observed_{year}_{base}'] = modeling[col].values

    X = build_anchor_features(modeling, anchor_year=inference_anchor_year)

    forecast_bundle = joblib.load(registry[f'tplus1_{final_model_name}']['path'])
    direct_bundle = joblib.load(registry[f'delta_{final_model_name}']['path'])

    forecast_bundle = joblib.load(registry[f'tplus1_{final_tplus1_model_name}']['path'])
    direct_bundle = joblib.load(registry[f'delta_{final_delta_model_name}']['path'])

    forecast_pred = forecast_bundle['pipeline'].predict(X[forecast_bundle['feature_columns']])
    forecast_df = pd.DataFrame(forecast_pred, columns=forecast_bundle['target_columns'], index=modeling.index)

    direct_pred = direct_bundle['pipeline'].predict(X[direct_bundle['feature_columns']])
    direct_df = pd.DataFrame(direct_pred, columns=direct_bundle['target_columns'], index=modeling.index)

    for base in LABEL_BASES:
        current = modeling[f'{base}_{inference_anchor_year}'].values
        out[f'forecast_{forecast_year}_{base}'] = forecast_df[f'next_{base}'].values
        out[f'derived_delta_{forecast_year}_{base}'] = out[f'forecast_{forecast_year}_{base}'] - current
        out[f'direct_delta_{forecast_year}_{base}'] = direct_df[f'delta_{base}'].values
        out[f'direct_reconstructed_{forecast_year}_{base}'] = current + out[f'direct_delta_{forecast_year}_{base}']
        out[f'ensemble_delta_{forecast_year}_{base}'] = (out[f'derived_delta_{forecast_year}_{base}'] + out[f'direct_delta_{forecast_year}_{base}']) / 2.0
        out[f'ensemble_{forecast_year}_{base}'] = current + out[f'ensemble_delta_{forecast_year}_{base}']

    ensemble_delta_cols = [f'ensemble_delta_{forecast_year}_{base}' for base in LABEL_BASES]
    out['ensemble_abs_total_change'] = out[ensemble_delta_cols].abs().sum(axis=1)
    out['ensemble_change_flag'] = (out['ensemble_abs_total_change'] >= float(cfg['modeling']['change_threshold'])).astype(int)
    out['ensemble_dominant_change_class'] = out[ensemble_delta_cols].abs().idxmax(axis=1).str.replace(f'ensemble_delta_{forecast_year}_', '', regex=False)

    out['forecast_anchor_year'] = inference_anchor_year
    out['forecast_year'] = forecast_year
    out['tplus1_model_name'] = final_tplus1_model_name
    out['delta_model_name'] = final_delta_model_name
    out['cell_size_m'] = float(cfg['grid']['cell_size_m'])
    out['intended_users'] = ', '.join(cfg['reporting'].get('intended_users', []))
    out['not_for_decisions'] = ', '.join(cfg['reporting'].get('not_for_decisions', []))

    unc_path = resolve_from_config(cfg, cfg['paths']['uncertainty_table'])
    if unc_path.exists():
        unc = pd.read_parquet(unc_path)
        if 'grid_id' in unc.columns:
            out = out.merge(unc, on='grid_id', how='left')

    out_gdf = grid.merge(out, on='grid_id', how='inner')
    ensure_parent(resolve_from_config(cfg, cfg['paths']['app_predictions']))
    out_gdf.to_file(resolve_from_config(cfg, cfg['paths']['app_predictions']), driver='GeoJSON')
    logger.info('Saved unified app predictions to %s', resolve_from_config(cfg, cfg['paths']['app_predictions']))


if __name__ == '__main__':
    main()
