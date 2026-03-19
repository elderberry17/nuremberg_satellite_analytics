import numpy as np
import pandas as pd
from pathlib import Path

from config import WC_GROUPS
from feature_engineering.build_tables import (
    DEFAULT_BANDS,
    GROUP_TO_ID,
    collect_dataset_items_tif,
    encode_time_features,
    extract_label_distribution,
    extract_window_features_multiband,
    load_feature_stack_from_tifs,
    load_label_tif,
    pad_array_reflect,
    read_single_band_tif,
    wc_label_to_group,
)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full_like(numerator, np.nan, dtype=np.float32)
    mask = denominator != 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def get_band_index_map(bands=DEFAULT_BANDS):
    return {band: i for i, band in enumerate(bands)}


def build_invalid_mask(
    feature_stack: np.ndarray,
    band_paths: dict,
    bands=DEFAULT_BANDS,
    skip_all_zero: bool = False,
):
    """
    Pixel-level invalid mask, shape (H, W).
    """
    invalid_mask = np.any(~np.isfinite(feature_stack), axis=-1)

    for i, band in enumerate(bands):
        _, _, nodata = read_single_band_tif(band_paths[band])
        if nodata is not None:
            invalid_mask |= (feature_stack[:, :, i] == nodata)

    if skip_all_zero:
        invalid_mask |= np.all(feature_stack == 0, axis=-1)

    return invalid_mask


def compute_window_spectral_features(
    window: np.ndarray,
    band_names=DEFAULT_BANDS,
    invalid_mask_window: np.ndarray | None = None,
):
    """
    window: (N, N, C)
    invalid_mask_window: (N, N), True for invalid pixels

    Returns NEW aggregated spectral features over the window.
    """
    band_to_idx = get_band_index_map(band_names)

    required = ["B04", "B08", "B11"]
    missing = [b for b in required if b not in band_to_idx]
    if missing:
        raise ValueError(f"Missing required bands for spectral features: {missing}")

    red = window[:, :, band_to_idx["B04"]].astype(np.float32)
    nir = window[:, :, band_to_idx["B08"]].astype(np.float32)
    swir = window[:, :, band_to_idx["B11"]].astype(np.float32)

    ndvi = safe_divide(nir - red, nir + red)
    ndbi = safe_divide(swir - nir, swir + nir)

    if invalid_mask_window is None:
        valid_mask = np.isfinite(red) & np.isfinite(nir) & np.isfinite(swir)
        cloud_pct = 0.0
    else:
        valid_mask = ~invalid_mask_window
        cloud_pct = 100.0 * invalid_mask_window.mean()

    valid_mask_ndvi = valid_mask & np.isfinite(ndvi)
    valid_mask_ndbi = valid_mask & np.isfinite(ndbi)

    feats = {
        "mean_NIR": float(np.nanmean(nir[valid_mask])) if np.any(valid_mask) else np.nan,
        "mean_SWIR": float(np.nanmean(swir[valid_mask])) if np.any(valid_mask) else np.nan,
        "mean_NDVI": float(np.nanmean(ndvi[valid_mask_ndvi])) if np.any(valid_mask_ndvi) else np.nan,
        "mean_NDBI": float(np.nanmean(ndbi[valid_mask_ndbi])) if np.any(valid_mask_ndbi) else np.nan,
        "std_NDVI": float(np.nanstd(ndvi[valid_mask_ndvi])) if np.any(valid_mask_ndvi) else np.nan,
        "cloud_pct": float(cloud_pct),
    }

    return feats


def image_tif_pair_to_table_extended(
    band_paths: dict,
    label_path: str,
    image_id: str,
    timestamp: pd.Timestamp,
    kernel_size: int = 3,
    keep_borders: bool = True,
    stride: int = 0,
    bands=DEFAULT_BANDS,
    skip_all_zero: bool = False,
):
    """
    Extended classification table:
    contains BOTH
      1. old baseline features from extract_window_features_multiband
      2. new spectral features
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    if stride < 0:
        raise ValueError("stride must be >= 0")

    step = 1 if stride in (0, 1) else stride

    feature_stack = load_feature_stack_from_tifs(band_paths, bands=bands)
    label_arr, _, label_nodata = load_label_tif(label_path)

    if feature_stack.shape[:2] != label_arr.shape:
        raise ValueError(
            f"Feature/label shape mismatch: {feature_stack.shape[:2]} vs {label_arr.shape}"
        )

    H, W, _ = feature_stack.shape
    radius = kernel_size // 2
    time_feats = encode_time_features(timestamp)

    invalid_mask = build_invalid_mask(
        feature_stack=feature_stack,
        band_paths=band_paths,
        bands=bands,
        skip_all_zero=skip_all_zero,
    )

    if keep_borders:
        feat_padded = pad_array_reflect(feature_stack, radius)
        invalid_padded = pad_array_reflect(invalid_mask.astype(np.uint8), radius).astype(bool)
        y_range = range(0, H, step)
        x_range = range(0, W, step)
    else:
        feat_padded = feature_stack
        invalid_padded = invalid_mask
        y_range = range(radius, H - radius, step)
        x_range = range(radius, W - radius, step)

    rows = []

    for y in y_range:
        for x in x_range:
            if invalid_mask[y, x]:
                continue

            label_value = int(label_arr[y, x])

            if label_nodata is not None and label_value == label_nodata:
                continue

            label_group = wc_label_to_group(label_value, unknown_value=None)
            if label_group is None:
                continue

            if keep_borders:
                yp = y + radius
                xp = x + radius

                feat_window = feat_padded[
                    yp - radius: yp + radius + 1,
                    xp - radius: xp + radius + 1,
                    :
                ]
                invalid_window = invalid_padded[
                    yp - radius: yp + radius + 1,
                    xp - radius: xp + radius + 1
                ]
            else:
                feat_window = feature_stack[
                    y - radius: y + radius + 1,
                    x - radius: x + radius + 1,
                    :
                ]
                invalid_window = invalid_mask[
                    y - radius: y + radius + 1,
                    x - radius: x + radius + 1
                ]

            # OLD features
            baseline_feats = extract_window_features_multiband(
                feat_window,
                band_names=bands,
            )

            # NEW features
            spectral_feats = compute_window_spectral_features(
                window=feat_window,
                band_names=bands,
                invalid_mask_window=invalid_window,
            )

            row = {
                "image_id": image_id,
                "label_path": label_path,
                "timestamp": timestamp.isoformat(),
                "x": x,
                "y": y,
                "x_norm": x / (W - 1) if W > 1 else 0.0,
                "y_norm": y / (H - 1) if H > 1 else 0.0,
                "label_value": label_value,
                "label_group": label_group,
                "label_group_id": GROUP_TO_ID[label_group],
                **time_feats,
                **baseline_feats,
                **spectral_feats,
            }

            rows.append(row)

    return pd.DataFrame(rows)


def image_tif_pair_to_table_distributed_extended(
    band_paths: dict,
    label_path: str,
    image_id: str,
    timestamp: pd.Timestamp,
    kernel_size: int = 3,
    keep_borders: bool = True,
    stride: int = 0,
    bands=DEFAULT_BANDS,
    skip_all_zero: bool = False,
):
    """
    Extended distributed table:
    contains BOTH
      1. old baseline features from extract_window_features_multiband
      2. new spectral features
      3. distribution labels
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    if stride < 0:
        raise ValueError("stride must be >= 0")

    step = 1 if stride in (0, 1) else stride

    feature_stack = load_feature_stack_from_tifs(band_paths, bands=bands)
    label_arr, _, label_nodata = load_label_tif(label_path)

    if feature_stack.shape[:2] != label_arr.shape:
        raise ValueError(
            f"Feature/label shape mismatch: {feature_stack.shape[:2]} vs {label_arr.shape}"
        )

    H, W, _ = feature_stack.shape
    radius = kernel_size // 2
    time_feats = encode_time_features(timestamp)

    invalid_mask = build_invalid_mask(
        feature_stack=feature_stack,
        band_paths=band_paths,
        bands=bands,
        skip_all_zero=skip_all_zero,
    )

    if keep_borders:
        feat_padded = pad_array_reflect(feature_stack, radius)
        label_padded = pad_array_reflect(label_arr, radius)
        invalid_padded = pad_array_reflect(invalid_mask.astype(np.uint8), radius).astype(bool)
        y_range = range(0, H, step)
        x_range = range(0, W, step)
    else:
        feat_padded = feature_stack
        label_padded = label_arr
        invalid_padded = invalid_mask
        y_range = range(radius, H - radius, step)
        x_range = range(radius, W - radius, step)

    rows = []

    for y in y_range:
        for x in x_range:
            if invalid_mask[y, x]:
                continue

            center_label = int(label_arr[y, x])
            if label_nodata is not None and center_label == label_nodata:
                continue

            if keep_borders:
                yp = y + radius
                xp = x + radius

                feat_window = feat_padded[
                    yp - radius: yp + radius + 1,
                    xp - radius: xp + radius + 1,
                    :
                ]
                label_window = label_padded[
                    yp - radius: yp + radius + 1,
                    xp - radius: xp + radius + 1
                ]
                invalid_window = invalid_padded[
                    yp - radius: yp + radius + 1,
                    xp - radius: xp + radius + 1
                ]
            else:
                feat_window = feature_stack[
                    y - radius: y + radius + 1,
                    x - radius: x + radius + 1,
                    :
                ]
                label_window = label_arr[
                    y - radius: y + radius + 1,
                    x - radius: x + radius + 1
                ]
                invalid_window = invalid_mask[
                    y - radius: y + radius + 1,
                    x - radius: x + radius + 1
                ]

            # OLD features
            baseline_feats = extract_window_features_multiband(
                feat_window,
                band_names=bands,
            )

            # NEW features
            spectral_feats = compute_window_spectral_features(
                window=feat_window,
                band_names=bands,
                invalid_mask_window=invalid_window,
            )

            # LABEL DISTRIBUTION
            dist = extract_label_distribution(label_window, WC_GROUPS)
            built_up_pct = 100.0 * dist["urban_prop"]

            row = {
                "image_id": image_id,
                "label_path": label_path,
                "timestamp": timestamp.isoformat(),
                "x": x,
                "y": y,
                "x_norm": x / (W - 1) if W > 1 else 0.0,
                "y_norm": y / (H - 1) if H > 1 else 0.0,
                **time_feats,
                **baseline_feats,
                **spectral_feats,
                **dist,
                "built_up_pct": float(built_up_pct),
            }

            rows.append(row)

    return pd.DataFrame(rows)


def build_split_table_tif_extended(
    items: list,
    kernel_size: int = 3,
    keep_borders: bool = True,
    stride: int = 0,
    distribution: bool = False,
    bands=DEFAULT_BANDS,
    skip_all_zero: bool = False,
):
    """
    One-pass builder with BOTH old and new features.
    """
    dfs = []

    for item in items:
        print(f"Processing {item['image_id']} ...")

        if distribution:
            df_img = image_tif_pair_to_table_distributed_extended(
                band_paths=item["band_paths"],
                label_path=item["label_path"],
                image_id=item["image_id"],
                timestamp=item["timestamp"],
                kernel_size=kernel_size,
                keep_borders=keep_borders,
                stride=stride,
                bands=bands,
                skip_all_zero=skip_all_zero,
            )
        else:
            df_img = image_tif_pair_to_table_extended(
                band_paths=item["band_paths"],
                label_path=item["label_path"],
                image_id=item["image_id"],
                timestamp=item["timestamp"],
                kernel_size=kernel_size,
                keep_borders=keep_borders,
                stride=stride,
                bands=bands,
                skip_all_zero=skip_all_zero,
            )

        dfs.append(df_img)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def merge_change_tables(
    df_t1: pd.DataFrame,
    df_t2: pd.DataFrame,
    suffix_t1: str = "20",
    suffix_t2: str = "21",
    keys: list[str] | None = None,
):
    """
    Merge two extended tables and compute change features.

    Keeps all columns from both sides with suffixes.
    """
    if keys is None:
        keys = ["x", "y"]

    left = df_t1.copy()
    right = df_t2.copy()

    left_rename = {c: f"{c}_{suffix_t1}" for c in left.columns if c not in keys}
    right_rename = {c: f"{c}_{suffix_t2}" for c in right.columns if c not in keys}

    left = left.rename(columns=left_rename)
    right = right.rename(columns=right_rename)

    merged = left.merge(right, on=keys, how="inner")

    # change features
    for base_name in ["mean_NDVI", "mean_NDBI", "mean_NIR", "mean_SWIR", "built_up_pct"]:
        c1 = f"{base_name}_{suffix_t1}"
        c2 = f"{base_name}_{suffix_t2}"
        if c1 in merged.columns and c2 in merged.columns:
            merged[f"delta_{base_name}"] = merged[c2] - merged[c1]

    return merged


def collect_items_extended(
    root: Path,
    folder_names: list[str],
    label_path: str | Path,
    bands=DEFAULT_BANDS,
):
    return collect_dataset_items_tif(
        root=root,
        folder_names=folder_names,
        label_path=label_path,
        bands=bands,
    )