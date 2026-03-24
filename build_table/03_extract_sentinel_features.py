from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject

from common import ensure_parent, get_logger, load_config, safe_ratio, stats_from_values

CONFIG_PATH = Path("../config/project_config.yaml")
logger = get_logger("03_extract_sentinel_features")

RGB_BANDS_10M = ["B02", "B03", "B04", "B08"]
SWIR_BANDS_20M = ["B11", "B12"]


def compute_indices(arr: np.ndarray, band_names: list[str]) -> dict[str, np.ndarray]:
    band_map = {name: arr[i] for i, name in enumerate(band_names)}
    indices: dict[str, np.ndarray] = {}

    if "B08" in band_map and "B04" in band_map:
        indices["ndvi"] = safe_ratio(band_map["B08"] - band_map["B04"], band_map["B08"] + band_map["B04"])

    if "B03" in band_map and "B08" in band_map:
        indices["ndwi"] = safe_ratio(band_map["B03"] - band_map["B08"], band_map["B03"] + band_map["B08"])

    if "B11" in band_map and "B08" in band_map:
        indices["ndbi"] = safe_ratio(band_map["B11"] - band_map["B08"], band_map["B11"] + band_map["B08"])

    return indices


def find_band_file(root: Path, band: str, resolution_folder: str) -> Path:
    folder = root / resolution_folder
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    matches = list(folder.glob(f"*_{band}_*.jp2"))
    if not matches:
        raise FileNotFoundError(f"Could not find JP2 for {band} in {folder}")
    if len(matches) > 1:
        logger.warning("Multiple matches for %s in %s. Using %s", band, folder, matches[0].name)

    return matches[0]


def read_band_window(src: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    return src.read(1, window=window).astype("float32")


def read_resampled_band_to_target(
    src: rasterio.io.DatasetReader,
    dst_transform,
    dst_crs,
    dst_width: int,
    dst_height: int,
) -> np.ndarray:
    dst = np.empty((dst_height, dst_width), dtype="float32")
    reproject(
        source=rasterio.band(src, 1),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_shape=(dst_width, dst_height),
        resampling=Resampling.bilinear,
    )
    return dst


def open_sentinel_jp2s(scene_root: str | Path):
    root = Path(scene_root)

    rgb_paths = {band: find_band_file(root, band, "R10m") for band in RGB_BANDS_10M}
    swir_paths = {band: find_band_file(root, band, "R20m") for band in SWIR_BANDS_20M}

    rgb_srcs = {band: rasterio.open(path) for band, path in rgb_paths.items()}
    swir_srcs = {band: rasterio.open(path) for band, path in swir_paths.items()}

    return rgb_srcs, swir_srcs


def close_sources(src_dict: dict[str, rasterio.io.DatasetReader]) -> None:
    for src in src_dict.values():
        src.close()


def extract_features_for_year(
    grid_gdf: gpd.GeoDataFrame,
    scene_root: str,
    year: int,
    cfg: dict,
) -> pd.DataFrame:
    s_cfg = cfg["sentinel"]
    stats = list(s_cfg.get("stats", ["median", "p25", "p75", "std"]))
    all_touched = bool(s_cfg.get("all_touched", False))
    nodata_values = set(s_cfg.get("nodata_values", [0]))
    use_swir = bool(s_cfg.get("use_swir_if_valid", True))

    rgb_srcs, swir_srcs = open_sentinel_jp2s(scene_root)

    try:
        ref_src = rgb_srcs["B02"]  # reference 10m grid
        grid_local = grid_gdf.to_crs(ref_src.crs)
        full_window = Window(0, 0, ref_src.width, ref_src.height)

        out: list[dict] = []
        logger.info("Extracting Sentinel JP2 features from %s for %d cells", scene_root, len(grid_local))

        for idx, row in enumerate(grid_local.itertuples(index=False), start=1):
            geom = row.geometry
            rec = {"grid_id": row.grid_id}

            window = from_bounds(*geom.bounds, transform=ref_src.transform).round_offsets().round_lengths()
            try:
                window = window.intersection(full_window)
            except Exception:
                out.append(rec)
                continue

            if int(window.width) < 1 or int(window.height) < 1:
                out.append(rec)
                continue

            rgb_arrays = []
            for band in RGB_BANDS_10M:
                rgb_arrays.append(read_band_window(rgb_srcs[band], window))
            rgb_arr = np.stack(rgb_arrays, axis=0)

            transform = ref_src.window_transform(window)
            if rgb_arr.shape[1] == 0 or rgb_arr.shape[2] == 0:
                out.append(rec)
                continue

            band_arrays = list(rgb_arr)
            band_names = RGB_BANDS_10M.copy()

            swir_used = 0
            if use_swir:
                try:
                    for band in SWIR_BANDS_20M:
                        swir_band = read_resampled_band_to_target(
                            swir_srcs[band],
                            dst_transform=transform,
                            dst_crs=ref_src.crs,
                            dst_width=rgb_arr.shape[2],
                            dst_height=rgb_arr.shape[1],
                        )
                        band_arrays.append(swir_band)
                        band_names.append(band)
                    swir_used = 1
                except Exception as e:
                    logger.warning("SWIR resampling failed for grid_id=%s: %s", row.grid_id, e)

            arr = np.stack(band_arrays, axis=0)

            try:
                mask = geometry_mask(
                    [geom],
                    transform=transform,
                    invert=True,
                    out_shape=(arr.shape[1], arr.shape[2]),
                    all_touched=all_touched,
                )
            except Exception:
                out.append(rec)
                continue

            if mask.sum() == 0:
                out.append(rec)
                continue

            for i, band_name in enumerate(band_names):
                vals = arr[i][mask]
                vals = vals[np.isfinite(vals)]
                if nodata_values:
                    vals = vals[~np.isin(vals, list(nodata_values))]
                band_stats = stats_from_values(vals, stats)
                for stat_name, value in band_stats.items():
                    rec[f"{band_name}_{stat_name}_{year}"] = value

            indices = compute_indices(arr, band_names)
            for idx_name, idx_arr in indices.items():
                idx_vals = idx_arr[mask]
                idx_vals = idx_vals[np.isfinite(idx_vals)]
                idx_stats = stats_from_values(idx_vals, stats)
                for stat_name, value in idx_stats.items():
                    rec[f"{idx_name}_{stat_name}_{year}"] = value

            rec[f"swir_used_{year}"] = swir_used
            out.append(rec)

            if idx % 200 == 0 or idx == len(grid_local):
                logger.info("Processed %d/%d cells", idx, len(grid_local))

        return pd.DataFrame(out)

    finally:
        close_sources(rgb_srcs)
        close_sources(swir_srcs)


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    grid = gpd.read_file(cfg["paths"]["grid_file"])

    feat20 = extract_features_for_year(
        grid,
        cfg["paths"]["sentinel_scene_2020"],
        2020,
        cfg,
    )
    feat21 = extract_features_for_year(
        grid,
        cfg["paths"]["sentinel_scene_2021"],
        2021,
        cfg,
    )

    features = feat20.merge(feat21, on="grid_id", how="inner")
    ensure_parent(cfg["paths"]["features_table"])
    features.to_parquet(cfg["paths"]["features_table"], index=False)
    logger.info("Saved features to %s with shape %s", cfg["paths"]["features_table"], features.shape)


if __name__ == "__main__":
    main()
