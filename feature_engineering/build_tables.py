import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt
import numpy as np
from config import DIRS2USE
from pathlib import Path
import pandas as pd
import re
import math


def parse_date_from_folder(folder_name: str):
    """
    Extract date like 20200327 from folder name.
    """
    m = re.search(r"_(\d{8})T\d{6}_", folder_name)
    if not m:
        raise ValueError(f"Could not parse date from folder name: {folder_name}")
    date_str = m.group(1)

    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    dt = pd.Timestamp(year=year, month=month, day=day)

    return dt


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


def find_image_paths(folder: Path):
    cropped = list(folder.glob("*_cropped.png"))
    aligned = list(folder.glob("*_label_aligned.png"))

    if len(cropped) != 1:
        raise ValueError(f"Expected exactly 1 cropped image in {folder}, got {len(cropped)}")
    if len(aligned) != 1:
        raise ValueError(f"Expected exactly 1 aligned label image in {folder}, got {len(aligned)}")
    
    return cropped[0], aligned[0]


def load_rgb_image(path: Path):
    """
    Reads image and keeps first 3 channels.
    Works for PNGs loaded by matplotlib.image.imread.
    """
    img = mpimg.imread(path)

    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape {img.shape} for {path}")

    if img.shape[2] < 3:
        raise ValueError(f"Expected at least 3 channels, got shape {img.shape} for {path}")

    img_rgb = img[:, :, :3]
    return img_rgb


def pad_image_reflect(img: np.ndarray, pad: int):
    if pad == 0:
        return img
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


def is_white_pixel(pixel: np.ndarray, threshold: float = 0.98) -> bool:
    """
    Identifies if a pixel is completely white (frame, clouds, etc)
    pixel: shape (3,)
    Works for float [0,1] and uint [0,255].
    """
    pixel = np.asarray(pixel)

    if np.issubdtype(pixel.dtype, np.integer):
        pixel = pixel.astype(np.float32) / 255.0

    return bool(np.all(pixel >= threshold))

def extract_window_features(window: np.ndarray):
    """
    window shape: (N, N, 3)
    Now works without Windows overlapping! 
    Returns simple baseline features.
    """
    feats = {}

    # center pixel
    center = window[window.shape[0] // 2, window.shape[1] // 2]
    feats["center_r"] = float(center[0])
    feats["center_g"] = float(center[1])
    feats["center_b"] = float(center[2])

    # channel-wise stats
    for ch_idx, ch_name in enumerate(["r", "g", "b"]):
        channel = window[:, :, ch_idx]
        feats[f"mean_{ch_name}"] = float(channel.mean())
        feats[f"std_{ch_name}"] = float(channel.std())
        feats[f"min_{ch_name}"] = float(channel.min())
        feats[f"max_{ch_name}"] = float(channel.max())

    return feats


def rgb_to_label_id(label_rgb: np.ndarray):
    """
    Convert RGB triplet into a stable string label.
    Good first baseline if labels are color-coded classes.
    """
    # If image values are float in [0,1], map to 0..255
    if np.issubdtype(label_rgb.dtype, np.floating):
        vals = np.round(label_rgb * 255).astype(int)
    else:
        vals = label_rgb.astype(int)

    return f"{vals[0]}_{vals[1]}_{vals[2]}"

def rgb_to_label_id(label_rgb: np.ndarray):
    """
    Convert RGB triplet into a stable string label.
    Good first baseline if labels are color-coded classes.
    """
    # If image values are float in [0,1], map to 0..255
    if np.issubdtype(label_rgb.dtype, np.floating):
        vals = np.round(label_rgb * 255).astype(int)
    else:
        vals = label_rgb.astype(int)

    return f"{vals[0]}_{vals[1]}_{vals[2]}"


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


def collect_dataset_items(root: Path, dirs2use: dict):
    """
    Returns:
      train_items -> list of dicts for 2020
      test_items  -> list of dicts for 2021
    """
    train_dirs = list(dirs2use.keys())
    test_dirs = list(dirs2use.values())

    train_items = []
    test_items = []

    for split_name, folder_names in [("train", train_dirs), ("test", test_dirs)]:
        for folder_name in folder_names:
            folder = root / folder_name
            cropped_path, aligned_path = find_image_paths(folder)
            dt = parse_date_from_folder(folder_name)

            item = {
                "folder_name": folder_name,
                "image_id": folder_name,
                "cropped_path": str(cropped_path),
                "aligned_path": str(aligned_path),
                "timestamp": dt,
            }

            if split_name == "train":
                train_items.append(item)
            else:
                test_items.append(item)

    return train_items, test_items


def image_pair_to_table(
    cropped_path: str,
    aligned_path: str,
    image_id: str,
    timestamp: pd.Timestamp,
    kernel_size: int = 3,
    keep_borders: bool = True,
    stride: int = 0,
):
    """
    One image pair -> DataFrame with one row per valid pixel.
    
    Parameters:
        kernel_size: size of NxN neighborhood
        keep_borders: whether to use reflected padding for border pixels
        stride:
            0 or 1 -> use all pixels
            k > 1  -> use every k-th pixel along x and y
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    if stride < 0:
        raise ValueError("stride must be >= 0")

    step = 1 if stride in (0, 1) else stride

    img = load_rgb_image(Path(cropped_path))
    label_img = load_rgb_image(Path(aligned_path))

    if img.shape[:2] != label_img.shape[:2]:
        raise ValueError(
            f"Image/label shape mismatch: {img.shape} vs {label_img.shape}"
        )

    H, W, _ = img.shape
    radius = kernel_size // 2
    time_feats = encode_time_features(timestamp)

    if keep_borders:
        img_padded = pad_image_reflect(img, radius)
        y_range = range(0, H, step)
        x_range = range(0, W, step)
    else:
        img_padded = img
        y_range = range(radius, H - radius, step)
        x_range = range(radius, W - radius, step)

    rows = []

    for y in y_range:
        for x in x_range:
            # skip white center pixels in raw image
            center_pixel = img[y, x, :]
            if is_white_pixel(center_pixel, threshold=0.98):
                continue

            if keep_borders:
                yp = y + radius
                xp = x + radius
                window = img_padded[
                    yp - radius: yp + radius + 1,
                    xp - radius: xp + radius + 1,
                    :
                ]
            else:
                window = img[
                    y - radius: y + radius + 1,
                    x - radius: x + radius + 1,
                    :
                ]

            feats = extract_window_features(window)

            label_rgb = label_img[y, x, :]
            label_id = rgb_to_label_id(label_rgb)

            row = {
                "image_id": image_id,
                "cropped_path": cropped_path,
                "aligned_path": aligned_path,
                "timestamp": timestamp.isoformat(),
                "x": x,
                "y": y,
                "x_norm": x / (W - 1) if W > 1 else 0.0,
                "y_norm": y / (H - 1) if H > 1 else 0.0,
                "label_r": float(label_rgb[0]),
                "label_g": float(label_rgb[1]),
                "label_b": float(label_rgb[2]),
                "label_id": label_id,
                **time_feats,
                **feats,
            }

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_split_table(
    items: list,
    kernel_size: int = 3,
    keep_borders: bool = True,
    stride: int = 0,
):
    """
    Takes items: [{folder_name: img_name}] to build the dataset from.
    
    Parameters:
        kernel_size: size of NxN neighborhood
        keep_borders: whether to process border pixels
        stride:
            0 or 1 -> use all pixels
            k > 1  -> use every k-th pixel
    """
    dfs = []

    for item in items:
        print(f"Processing {item['image_id']} ...")
        df_img = image_pair_to_table(
            cropped_path=item["cropped_path"],
            aligned_path=item["aligned_path"],
            image_id=item["image_id"],
            timestamp=item["timestamp"],
            kernel_size=kernel_size,
            keep_borders=keep_borders,
            stride=stride,
        )
        dfs.append(df_img)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)