from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

from common import ensure_parent, get_logger, load_config

CONFIG_PATH = Path('../config/project_config.yaml')
logger = get_logger('02_prepare_worldcover_labels')


def build_class_lookup(classes_cfg: dict) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for cls_name, spec in classes_cfg.items():
        for code in spec['worldcover_codes']:
            lookup[int(code)] = cls_name
    return lookup


def compute_yearly_proportions(grid_gdf: gpd.GeoDataFrame, raster_path: str, classes_cfg: dict, all_touched: bool, nodata_values: set[int | float]) -> pd.DataFrame:
    class_lookup = build_class_lookup(classes_cfg)
    class_names = list(classes_cfg.keys())

    with rasterio.open(raster_path) as src:
        grid_local = grid_gdf.to_crs(src.crs)
        full_window = Window(0, 0, src.width, src.height)
        records: list[dict] = []
        logger.info('Computing proportions from %s for %d cells', raster_path, len(grid_local))

        for idx, row in enumerate(grid_local.itertuples(index=False), start=1):
            geom = row.geometry
            rec = {'grid_id': row.grid_id}
            for c in class_names:
                rec[c] = np.nan

            window = from_bounds(*geom.bounds, transform=src.transform).round_offsets().round_lengths()
            try:
                window = window.intersection(full_window)
            except Exception:
                records.append(rec)
                continue

            if int(window.width) < 1 or int(window.height) < 1:
                records.append(rec)
                continue

            arr = src.read(1, window=window)
            transform = src.window_transform(window)
            try:
                mask = geometry_mask([geom], transform=transform, invert=True, out_shape=arr.shape, all_touched=all_touched)
            except Exception:
                records.append(rec)
                continue

            values = arr[mask]
            values = values[np.isfinite(values)]
            if nodata_values:
                values = values[~np.isin(values, list(nodata_values))]
            if values.size == 0:
                records.append(rec)
                continue

            mapped = np.array([class_lookup.get(int(v), 'other') for v in values], dtype=object)
            total = max(mapped.size, 1)
            for c in class_names:
                rec[c] = float(np.sum(mapped == c) / total)
            records.append(rec)

            if idx % 200 == 0 or idx == len(grid_local):
                logger.info('Processed %d/%d cells', idx, len(grid_local))

    return pd.DataFrame(records)


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    grid = gpd.read_file(cfg['paths']['grid_file'])
    classes_cfg = cfg['classes']
    wc_cfg = cfg['worldcover']
    nodata = set(wc_cfg.get('nodata_values', [0]))

    wc20 = compute_yearly_proportions(grid, cfg['paths']['worldcover_2020'], classes_cfg, bool(wc_cfg.get('all_touched', False)), nodata)
    wc21 = compute_yearly_proportions(grid, cfg['paths']['worldcover_2021'], classes_cfg, bool(wc_cfg.get('all_touched', False)), nodata)

    wc20 = wc20.rename(columns={c: f'{c}_2020' for c in classes_cfg.keys()})
    wc21 = wc21.rename(columns={c: f'{c}_2021' for c in classes_cfg.keys()})
    labels = wc20.merge(wc21, on='grid_id', how='inner')

    for c in classes_cfg.keys():
        labels[f'delta_{c}'] = labels[f'{c}_2021'] - labels[f'{c}_2020']
    labels['class_sum_2020'] = labels[[f'{c}_2020' for c in classes_cfg.keys()]].sum(axis=1)
    labels['class_sum_2021'] = labels[[f'{c}_2021' for c in classes_cfg.keys()]].sum(axis=1)

    ensure_parent(cfg['paths']['labels_table'])
    labels.to_parquet(cfg['paths']['labels_table'], index=False)
    logger.info('Saved labels to %s with shape %s', cfg['paths']['labels_table'], labels.shape)


if __name__ == '__main__':
    main()
