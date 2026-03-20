import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from config import WC_GROUPS


DEFAULT_BANDS = ["B02", "B03", "B04", "B08", "B11"]

GROUP_TO_ID = {
    "urban": 0,
    "water": 1,
    "vegetation": 2,
}


def build_wc_value_to_group(groups: dict[str, set[int]]):
    """
    Example:
        {
            "urban": {50},
            "water": {80, 90, 95},
            "vegetation": {10, 20, 30, 40, 60, 100},
        }
    ->
        {
            50: "urban",
            80: "water",
            ...
        }
    """
    mapping = {}

    for group_name, class_values in groups.items():
        for value in class_values:
            if value in mapping:
                raise ValueError(f"Class value {value} is assigned to multiple groups")
            mapping[int(value)] = group_name

    return mapping


WC_VALUE_TO_GROUP = build_wc_value_to_group(WC_GROUPS)


def wc_label_to_group(label_value: int, unknown_value=None):
    """
    Map raw WorldCover class -> collapsed group.
    Returns one of:
        "urban", "water", "vegetation"
    or unknown_value if class is not covered by WC_GROUPS.
    """
    return WC_VALUE_TO_GROUP.get(int(label_value), unknown_value)


def parse_date_from_folder(folder_name: str):
    """
    Extract date like 20200327 from folder name.

    Example folder:
    urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20200713T103031_32UPV_TOC_V210__20200713T103031
    """
    m = re.search(r"_(\d{8})T\d{6}", folder_name)
    if not m:
        raise ValueError(f"Could not parse date from folder name: {folder_name}")

    date_str = m.group(1)

    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    return pd.Timestamp(year=year, month=month, day=day)


def encode_time_features(dt: pd.Timestamp):
    """
    Basic time features for classical ML.
    """
    day_of_year = dt.dayofyear
    days_in_year = 366 if dt.is_leap_year else 365

    doy_sin = math.sin(2 * math.pi * day_of_year / days_in_year)
    doy_cos = math.cos(2 * math.pi * day_of_year / days_in_year)

    return {
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "day_of_year": day_of_year,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
    }


def find_band_paths(folder: Path, bands=DEFAULT_BANDS):
    """
    Finds one .tif path per band in the given folder.

    More robust than direct glob-by-pattern:
    1. list all tif files in folder
    2. match by filename substring
    """
    folder = Path(folder)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")

    all_tifs = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".tif"])

    if not all_tifs:
        raise ValueError(f"No .tif files found in folder: {folder}")

    band_paths = {}

    for band in bands:
        matches = [p for p in all_tifs if band in p.name and "aligned" in p.name]

        if len(matches) != 1:
            available = [p.name for p in all_tifs]
            raise ValueError(
                f"Expected exactly 1 tif for band {band} in {folder}, got {len(matches)}.\n"
                f"Matches: {[p.name for p in matches]}\n"
                f"Available tif files: {available}"
            )

        band_paths[band] = matches[0]

    return band_paths


def read_single_band_tif(path: str | Path):
    """
    Reads a single-band tif and returns:
      data: np.ndarray of shape (H, W)
      profile: raster profile
      nodata: nodata value
    """
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile
        nodata = src.nodata

    return data, profile, nodata


def load_feature_stack_from_tifs(
    band_paths: dict,
    bands=DEFAULT_BANDS,
    dtype=np.float32,
):
    """
    Reads multiple single-band tif files and stacks them into (H, W, C).
    """
    arrays = []
    ref_shape = None

    for band in bands:
        path = band_paths[band]
        arr, _, _ = read_single_band_tif(path)

        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(
                f"Band shape mismatch for {band}: got {arr.shape}, expected {ref_shape}"
            )

        arrays.append(arr.astype(dtype))

    stacked = np.stack(arrays, axis=-1)  # (H, W, C)
    return stacked


def load_label_tif(path: str | Path):
    """
    Reads label tif as (H, W).
    """
    label_arr, profile, nodata = read_single_band_tif(path)
    return label_arr, profile, nodata


def pad_array_reflect(arr: np.ndarray, pad: int):
    if pad == 0:
        return arr

    if arr.ndim == 2:
        return np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
    elif arr.ndim == 3:
        return np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    else:
        raise ValueError(f"Unsupported ndim={arr.ndim}")


def is_invalid_center_pixel(
    feature_stack: np.ndarray,
    y: int,
    x: int,
    nodata_values: dict | None = None,
    bands=DEFAULT_BANDS,
    skip_all_zero: bool = False,
):
    """
    Skip center pixel if:
      - all channels are zero and skip_all_zero=True
      - any channel equals nodata
      - any channel is not finite
    """
    center = feature_stack[y, x, :]  # shape: (C,)

    if skip_all_zero and np.all(center == 0):
        return True

    if nodata_values is not None:
        for i, band in enumerate(bands):
            nd = nodata_values.get(band, None)
            if nd is not None and center[i] == nd:
                return True

    if np.any(~np.isfinite(center)):
        return True

    return False


def extract_window_features_multiband(
    window: np.ndarray,
    band_names=DEFAULT_BANDS,
):
    """
    window shape: (N, N, C)
    Returns simple baseline features per band.
    """
    feats = {}

    cy = window.shape[0] // 2
    cx = window.shape[1] // 2
    center = window[cy, cx, :]

    for ch_idx, band_name in enumerate(band_names):
        band_window = window[:, :, ch_idx]
        band_name_lower = band_name.lower()

        feats[f"center_{band_name_lower}"] = float(center[ch_idx])
        feats[f"mean_{band_name_lower}"] = float(band_window.mean())
        feats[f"std_{band_name_lower}"] = float(band_window.std())
        feats[f"min_{band_name_lower}"] = float(band_window.min())
        feats[f"max_{band_name_lower}"] = float(band_window.max())

    return feats


def extract_label_distribution(label_window: np.ndarray, wc_groups: dict[str, set[int]]):
    """
    Compute label proportions in the window after collapsing to 3 groups.

    Note:
    - classes outside WC_GROUPS are ignored
    - so urban_prop + water_prop + vegetation_prop may be < 1.0
    """
    total_pixels = label_window.size
    flat = label_window.reshape(-1).astype(int)

    urban = np.isin(flat, list(wc_groups["urban"])).sum()
    water = np.isin(flat, list(wc_groups["water"])).sum()
    vegetation = np.isin(flat, list(wc_groups["vegetation"])).sum()

    return {
        "urban_prop": urban / total_pixels,
        "water_prop": water / total_pixels,
        "vegetation_prop": vegetation / total_pixels,
    }


def estimate_num_rows(
    num_images: int,
    height: int,
    width: int,
    kernel_size: int = 3,
    keep_borders: bool = True,
    stride: int = 0,
) -> int:
    """
    Estimates the potential number of rows in a dataset.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    if stride < 0:
        raise ValueError("stride must be >= 0")

    step = 1 if stride in (0, 1) else stride
    radius = kernel_size // 2

    if keep_borders:
        effective_h = height
        effective_w = width
    else:
        effective_h = max(0, height - 2 * radius)
        effective_w = max(0, width - 2 * radius)

    n_y = math.ceil(effective_h / step)
    n_x = math.ceil(effective_w / step)

    return num_images * n_y * n_x


def collect_dataset_items_tif(
    root: Path,
    folder_names: list[str],
    label_path: str | Path,
    bands=DEFAULT_BANDS,
):
    """
    Returns:
      items -> list of dicts, one per timestamp folder
    """
    items = []
    label_path = str(label_path)

    for folder_name in folder_names:
        folder = root / folder_name
        band_paths = find_band_paths(folder, bands=bands)
        dt = parse_date_from_folder(folder_name)

        item = {
            "folder_name": folder_name,
            "image_id": folder_name,
            "folder_path": str(folder),
            "label_path": label_path,
            "timestamp": dt,
            "band_paths": {k: str(v) for k, v in band_paths.items()},
        }
        items.append(item)

    return items


def collect_dataset_items_tif_multilabel(
    root: Path,
    folder_names: list[str],
    label_paths: dict[str, str | Path],
    bands=DEFAULT_BANDS,
):
    """
    Returns:
      items -> list of dicts, one per timestamp folder

    label_paths example:
        {
            "2020": "/path/to/wc_2020.tif",
            "2021": "/path/to/wc_2021.tif",
        }
    """
    items = []

    label_paths = {str(k): str(v) for k, v in label_paths.items()}

    for folder_name in folder_names:
        folder = root / folder_name
        band_paths = find_band_paths(folder, bands=bands)
        dt = parse_date_from_folder(folder_name)

        item = {
            "folder_name": folder_name,
            "image_id": folder_name,
            "folder_path": str(folder),
            "label_paths": label_paths,
            "timestamp": dt,
            "band_paths": {k: str(v) for k, v in band_paths.items()},
        }
        items.append(item)

    return items


def image_tif_pair_to_table(
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
    One tif multiband sample -> DataFrame with one row per valid pixel.

    Target:
      label_value     raw WorldCover value
      label_group     urban / water / vegetation
      label_group_id  0 / 1 / 2

    Pixels with label classes outside WC_GROUPS are skipped.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    if stride < 0:
        raise ValueError("stride must be >= 0")

    step = 1 if stride in (0, 1) else stride

    feature_stack = load_feature_stack_from_tifs(band_paths, bands=bands)
    H, W, _ = feature_stack.shape

    nodata_values = {}
    for band in bands:
        _, _, nd = read_single_band_tif(band_paths[band])
        nodata_values[band] = nd

    label_arr, _, label_nodata = load_label_tif(label_path)

    if label_arr.shape != (H, W):
        raise ValueError(
            f"Feature/label shape mismatch: features {(H, W)} vs label {label_arr.shape}"
        )

    radius = kernel_size // 2
    time_feats = encode_time_features(timestamp)

    if keep_borders:
        feat_padded = pad_array_reflect(feature_stack, radius)
        y_range = range(0, H, step)
        x_range = range(0, W, step)
    else:
        feat_padded = feature_stack
        y_range = range(radius, H - radius, step)
        x_range = range(radius, W - radius, step)

    rows = []

    for y in y_range:
        for x in x_range:
            if is_invalid_center_pixel(
                feature_stack=feature_stack,
                y=y,
                x=x,
                nodata_values=nodata_values,
                bands=bands,
                skip_all_zero=skip_all_zero,
            ):
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
                window = feat_padded[
                    yp - radius: yp + radius + 1,
                    xp - radius: xp + radius + 1,
                    :
                ]
            else:
                window = feature_stack[
                    y - radius: y + radius + 1,
                    x - radius: x + radius + 1,
                    :
                ]

            feats = extract_window_features_multiband(window, band_names=bands)

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
                **feats,
            }

            rows.append(row)

    return pd.DataFrame(rows)


def image_tif_pair_to_table_distributed(
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
    One tif multiband sample -> DataFrame with one row per valid pixel.

    Target:
      urban_prop
      water_prop
      vegetation_prop

    Proportions are computed from label window using WC_GROUPS.
    Classes outside WC_GROUPS are ignored, so proportions may sum to < 1.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    if stride < 0:
        raise ValueError("stride must be >= 0")

    step = 1 if stride in (0, 1) else stride

    feature_stack = load_feature_stack_from_tifs(band_paths, bands=bands)
    H, W, _ = feature_stack.shape

    nodata_values = {}
    for band in bands:
        _, _, nd = read_single_band_tif(band_paths[band])
        nodata_values[band] = nd

    label_arr, _, label_nodata = load_label_tif(label_path)

    if label_arr.shape != (H, W):
        raise ValueError(
            f"Feature/label shape mismatch: features {(H, W)} vs label {label_arr.shape}"
        )

    radius = kernel_size // 2
    time_feats = encode_time_features(timestamp)

    if keep_borders:
        feat_padded = pad_array_reflect(feature_stack, radius)
        label_padded = pad_array_reflect(label_arr, radius)
        y_range = range(0, H, step)
        x_range = range(0, W, step)
    else:
        feat_padded = feature_stack
        label_padded = label_arr
        y_range = range(radius, H - radius, step)
        x_range = range(radius, W - radius, step)

    rows = []

    for y in y_range:
        for x in x_range:
            if is_invalid_center_pixel(
                feature_stack=feature_stack,
                y=y,
                x=x,
                nodata_values=nodata_values,
                bands=bands,
                skip_all_zero=skip_all_zero,
            ):
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

            feats = extract_window_features_multiband(feat_window, band_names=bands)
            dist = extract_label_distribution(label_window, WC_GROUPS)

            row = {
                "image_id": image_id,
                "label_path": label_path,
                "timestamp": timestamp.isoformat(),
                "x": x,
                "y": y,
                "x_norm": x / (W - 1) if W > 1 else 0.0,
                "y_norm": y / (H - 1) if H > 1 else 0.0,
                **time_feats,
                **feats,
                **dist,
            }

            rows.append(row)

    return pd.DataFrame(rows)


def build_split_table_tif(
    items: list,
    kernel_size: int = 3,
    keep_borders: bool = True,
    stride: int = 0,
    distribution: bool = False,
    bands=DEFAULT_BANDS,
    skip_all_zero: bool = False,
):
    """
    Build one big dataframe from collected tif items.

    Parameters:
        items: list of dicts from collect_dataset_items_tif(...)
        kernel_size: size of NxN neighborhood
        keep_borders: if True, use reflected padding for border pixels
        stride:
            0 or 1 -> use all pixels
            k > 1  -> use every k-th pixel
        distribution:
            False -> classification mode
            True  -> distribution mode
    """
    dfs = []

    for item in items:
        print(f"Processing {item['image_id']} ...")

        if distribution:
            df_img = image_tif_pair_to_table_distributed(
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
            df_img = image_tif_pair_to_table(
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
def extract_window_distribution(window: np.ndarray):
    h, w, _ = window.shape
    total_pixels = h * w
    
    # Flatten pixels and map them into 0-255
    pixels = (window.reshape(-1, 3) * 255).astype(int)
    rgb_strings = np.array([f"{r}_{g}_{b}" for r, g, b in pixels])
    labels = np.array([RGB_STR2LABEL_STR.get(k, "unknown") for k in rgb_strings])
    
    water = np.isin(labels, GROUPS["water"]).sum()
    urban = np.isin(labels, GROUPS["urban"]).sum()
    vegetation = np.isin(labels, GROUPS["vegetation"]).sum()

    return pd.concat(dfs, ignore_index=True)
