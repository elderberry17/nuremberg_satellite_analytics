from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import yaml
from shapely.geometry import box


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('_config_path', str(config_path.resolve()))
    cfg.setdefault('_config_dir', str(config_path.resolve().parent))
    return cfg


def resolve_from_config(cfg: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(cfg.get('_config_dir', '.'))
    return (base / path).resolve()


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    return logging.getLogger(name)


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype='float32')
    den = np.asarray(den, dtype='float32')
    out = np.full_like(num, np.nan, dtype='float32')
    valid = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[valid] = num[valid] / den[valid]
    return out


def clean_polygon_gdf(gdf: gpd.GeoDataFrame, metric_crs: str) -> gpd.GeoDataFrame:
    if gdf.empty:
        raise ValueError('GeoDataFrame is empty.')
    if gdf.crs is None:
        raise ValueError('GeoDataFrame has no CRS.')
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    if gdf.empty:
        raise ValueError('No polygon geometries found after cleaning.')
    gdf = gdf.to_crs(metric_crs)
    gdf = gdf.dissolve().reset_index(drop=True)
    gdf['geometry'] = gdf.geometry.buffer(0)
    return gdf


def load_boundary(path: str | Path, metric_crs: str) -> gpd.GeoDataFrame:
    return clean_polygon_gdf(gpd.read_file(path), metric_crs)


def make_regular_grid(boundary_gdf: gpd.GeoDataFrame, cell_size_m: float, keep_full_cells: bool = True) -> gpd.GeoDataFrame:
    boundary_geom = boundary_gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    xs = np.arange(minx, maxx + cell_size_m, cell_size_m)
    ys = np.arange(miny, maxy + cell_size_m, cell_size_m)

    polys = []
    gids = []
    gid = 0
    for x in xs[:-1]:
        for y in ys[:-1]:
            polys.append(box(x, y, x + cell_size_m, y + cell_size_m))
            gids.append(gid)
            gid += 1

    grid = gpd.GeoDataFrame({'grid_id': gids}, geometry=polys, crs=boundary_gdf.crs)
    if keep_full_cells:
        grid = grid.loc[grid.intersects(boundary_geom)].copy().reset_index(drop=True)
    else:
        grid = gpd.overlay(grid, boundary_gdf[['geometry']], how='intersection')
        grid = grid.loc[~grid.geometry.is_empty].copy().reset_index(drop=True)
    grid['grid_id'] = np.arange(len(grid))
    grid['area_m2'] = grid.geometry.area
    return grid


def stats_from_values(values: np.ndarray, stats: Sequence[str]) -> dict[str, float]:
    vals = np.asarray(values, dtype='float32')
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {s: np.nan for s in stats}
    out: dict[str, float] = {}
    if 'median' in stats:
        out['median'] = float(np.nanmedian(vals))
    if 'p25' in stats:
        out['p25'] = float(np.nanpercentile(vals, 25))
    if 'p75' in stats:
        out['p75'] = float(np.nanpercentile(vals, 75))
    if 'std' in stats:
        out['std'] = float(np.nanstd(vals))
    if 'mean' in stats:
        out['mean'] = float(np.nanmean(vals))
    return out


def dump_json(path: str | Path, payload: dict) -> None:
    ensure_parent(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
