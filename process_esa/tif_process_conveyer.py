import os
import re
import glob
import csv
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import Affine
from shapely.geometry import Polygon, mapping
import pyproj
from shapely.ops import transform as shp_transform
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
INPUT_DIR = "S2_2020_downloads/S2A_20200302T102021_32UPV_TOC_V210"
OUT_DIR = "S2_2020_processed"

# ROI in lon/lat (EPSG:4326) — как у вас в примерах
ROI_COORDS = [
    (10.973282, 49.556621),
    (11.183739, 49.556621),
    (11.183739, 49.344138),
    (10.973282, 49.344138),
    (10.973282, 49.556621),
]
ROI_POLY_WGS84 = Polygon(ROI_COORDS)

# Sentinel-2 TOC reflectance often uses 0..10000-ish (depends on product)
# For visualization:
CLIP_MAX = 3000.0

SAVE_CROPPED_GEOTIFF = True  # if you want a cropped 3-band tif too

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
SCENE_RE = re.compile(r"^(S2[AB]_\d{8}T\d{6}_\d{2}[A-Z]{3})_")  # e.g. S2A_20200302T102021_32UPV_
TS_RE = re.compile(r"_(\d{8}T\d{6})_")  # extracts 20200302T102021


def find_scenes(input_dir: str):
    """
    Group band files by scene_id prefix like:
    S2A_20200302T102021_32UPV
    """
    all_tifs = glob.glob(os.path.join(input_dir, "*.tif"))
    scenes = {}
    for p in all_tifs:
        base = os.path.basename(p)
        m = SCENE_RE.match(base)
        if not m:
            continue
        scene_id = m.group(1)
        scenes.setdefault(scene_id, []).append(p)
    return scenes


def get_band_path(scene_files, band_token: str):
    """
    band_token examples:
      "TOC-B02_10M"
      "TOC-B03_10M"
      "TOC-B04_10M"
    """
    for p in scene_files:
        if band_token in os.path.basename(p):
            return p
    return None


def wgs84_polygon_to_raster_crs(poly_wgs84: Polygon, raster_crs):
    """Reproject ROI polygon from EPSG:4326 to raster CRS."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    return shp_transform(transformer.transform, poly_wgs84)


def crop_single_band(path: str, roi_poly_wgs84: Polygon):
    """
    Returns:
      band (H,W) float32
      out_transform
      out_crs
      out_meta_base (for writing)
    """
    with rasterio.open(path) as src:
        roi_poly_proj = wgs84_polygon_to_raster_crs(roi_poly_wgs84, src.crs)
        arr, out_transform = mask(src, [mapping(roi_poly_proj)], crop=True)
        # arr shape: (1, H, W)
        band = arr[0].astype(np.float32)
        meta = src.meta.copy()
        meta.update({
            "height": band.shape[0],
            "width": band.shape[1],
            "transform": out_transform,
            "count": 1,
            "dtype": "float32",
        })
        return band, out_transform, src.crs, meta


def save_rgb_png(rgb_float01: np.ndarray, out_path: str):
    # rgb_float01 expected shape (H,W,3), values in [0,1]
    plt.imsave(out_path, rgb_float01)


def save_rgb_geotiff(rgb_float32: np.ndarray, out_path: str, crs, transform: Affine):
    # rgb_float32 shape: (H,W,3) float32
    h, w, _ = rgb_float32.shape
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 3,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(out_path, "w", **meta) as dst:
        # rasterio wants (bands, H, W)
        dst.write(rgb_float32[:, :, 0], 1)  # R
        dst.write(rgb_float32[:, :, 1], 2)  # G
        dst.write(rgb_float32[:, :, 2], 3)  # B


def extract_timestamp(scene_id: str) -> str:
    """
    scene_id example: S2A_20200302T102021_32UPV
    Returns timestamp string: 20200302T102021
    """
    m = TS_RE.search("_" + scene_id + "_")
    if not m:
        # fallback: try another strategy
        m2 = re.search(r"_(\d{8}T\d{6})_", scene_id + "_")
        if not m2:
            return "UNKNOWN_TS"
        return m2.group(1)
    return m.group(1)


# -----------------------
# Main
# -----------------------
scenes = find_scenes(INPUT_DIR)
print(f"Found scenes: {len(scenes)}")

index_rows = []

for scene_id, files in sorted(scenes.items()):
    ts = extract_timestamp(scene_id)

    b02_path = get_band_path(files, "TOC-B02_10M")  # Blue
    b03_path = get_band_path(files, "TOC-B03_10M")  # Green
    b04_path = get_band_path(files, "TOC-B04_10M")  # Red

    if not (b02_path and b03_path and b04_path):
        print(f"[SKIP] {scene_id}: missing one of B02/B03/B04 10m")
        continue

    # Crop each band to ROI
    red, out_transform, out_crs, _ = crop_single_band(b04_path, ROI_POLY_WGS84)
    green, _, _, _ = crop_single_band(b03_path, ROI_POLY_WGS84)
    blue, _, _, _ = crop_single_band(b02_path, ROI_POLY_WGS84)

    # Stack RGB
    rgb = np.dstack((red, green, blue)).astype(np.float32)

    # Simple visualization stretch to [0,1]
    rgb_vis = np.clip(rgb, 0, CLIP_MAX) / CLIP_MAX

    # Output paths (timestamp is mandatory)
    out_png = os.path.join(OUT_DIR, f"{scene_id}__{ts}__RGB_cropped.png")

    save_rgb_png(rgb_vis, out_png)

    out_tif = ""
    if SAVE_CROPPED_GEOTIFF:
        out_tif = os.path.join(OUT_DIR, f"{scene_id}__{ts}__RGB_cropped.tif")
        save_rgb_geotiff(rgb, out_tif, out_crs, out_transform)

    print(f"[OK] {scene_id} -> {out_png}")

    index_rows.append({
        "scene_id": scene_id,
        "timestamp": ts,
        "b04_red_path": os.path.basename(b04_path),
        "b03_green_path": os.path.basename(b03_path),
        "b02_blue_path": os.path.basename(b02_path),
        "out_png": os.path.basename(out_png),
        "out_tif": os.path.basename(out_tif) if out_tif else "",
    })

# Write index CSV for bookkeeping
index_csv = os.path.join(OUT_DIR, "index.csv")
with open(index_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=index_rows[0].keys() if index_rows else [
        "scene_id","timestamp","b04_red_path","b03_green_path","b02_blue_path","out_png","out_tif"
    ])
    w.writeheader()
    for r in index_rows:
        w.writerow(r)

print(f"Done. Wrote {len(index_rows)} rows to {index_csv}")