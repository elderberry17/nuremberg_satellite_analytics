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
CLIP_MAX = 3000.0
SAVE_CROPPED_GEOTIFF = True

ROI_COORDS = [
    (10.973282, 49.556621),
    (11.183739, 49.556621),
    (11.183739, 49.344138),
    (10.973282, 49.344138),
    (10.973282, 49.556621),
]
ROI_POLY_WGS84 = Polygon(ROI_COORDS)

# Extract timestamp from folder name like "...__20210908T101559"
FOLDER_TS_RE = re.compile(r"__(\d{8}T\d{6})$")

# -----------------------
# Helpers
# -----------------------
def get_band_path(scene_dir: str, band_token: str) -> str | None:
    # match both .tif and .jp2 just in case
    candidates = glob.glob(os.path.join(scene_dir, f"*{band_token}*.tif")) + \
                 glob.glob(os.path.join(scene_dir, f"*{band_token}*.jp2"))
    return candidates[0] if candidates else None


def wgs84_polygon_to_raster_crs(poly_wgs84: Polygon, raster_crs):
    transformer = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    return shp_transform(transformer.transform, poly_wgs84)


def crop_single_band(path: str, roi_poly_wgs84: Polygon):
    with rasterio.open(path) as src:
        roi_poly_proj = wgs84_polygon_to_raster_crs(roi_poly_wgs84, src.crs)
        arr, out_transform = mask(src, [mapping(roi_poly_proj)], crop=True)
        band = arr[0].astype(np.float32)
        return band, out_transform, src.crs


def save_rgb_png(rgb_float01: np.ndarray, out_path: str):
    plt.imsave(out_path, rgb_float01)


def save_rgb_geotiff(rgb_float32: np.ndarray, out_path: str, crs, transform: Affine):
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
        dst.write(rgb_float32[:, :, 0], 1)  # R
        dst.write(rgb_float32[:, :, 1], 2)  # G
        dst.write(rgb_float32[:, :, 2], 3)  # B


def extract_timestamp_from_folder(folder_name: str) -> str:
    m = FOLDER_TS_RE.search(folder_name)
    return m.group(1) if m else "UNKNOWN_TS"


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    ROOT_DIR = "S2_RGB_no_clouds"   # contains many scene folders

    scene_dirs = sorted([p for p in glob.glob(os.path.join(ROOT_DIR, "*")) if os.path.isdir(p)])
    print("Found scene folders:", len(scene_dirs))

    index_rows = []

    for scene_dir in scene_dirs:
        folder_name = os.path.basename(scene_dir)
        ts = extract_timestamp_from_folder(folder_name)

        b02_path = get_band_path(scene_dir, "TOC-B02_10M")  # Blue
        b03_path = get_band_path(scene_dir, "TOC-B03_10M")  # Green
        b04_path = get_band_path(scene_dir, "TOC-B04_10M")  # Red

        if not (b02_path and b03_path and b04_path):
            print(f"[SKIP] {folder_name}: missing B02/B03/B04 10m")
            continue

        # Crop each band to ROI
        red, out_transform, out_crs = crop_single_band(b04_path, ROI_POLY_WGS84)
        green, _, _ = crop_single_band(b03_path, ROI_POLY_WGS84)
        blue, _, _ = crop_single_band(b02_path, ROI_POLY_WGS84)

        rgb = np.dstack((red, green, blue)).astype(np.float32)
        rgb_vis = np.clip(rgb, 0, CLIP_MAX) / CLIP_MAX

        # Save inside the same folder
        out_png = os.path.join(scene_dir, f"{folder_name}__RGB_cropped.png")
        save_rgb_png(rgb_vis, out_png)

        out_tif = ""
        if SAVE_CROPPED_GEOTIFF:
            out_tif = os.path.join(scene_dir, f"{folder_name}__RGB_cropped.tif")
            save_rgb_geotiff(rgb, out_tif, out_crs, out_transform)

        print(f"[OK] {folder_name} -> {out_png}")

        index_rows.append({
            "scene_folder": folder_name,
            "timestamp": ts,
            "b04_red_path": os.path.basename(b04_path),
            "b03_green_path": os.path.basename(b03_path),
            "b02_blue_path": os.path.basename(b02_path),
            "out_png": os.path.basename(out_png),
            "out_tif": os.path.basename(out_tif) if out_tif else "",
        })

    # Save index.csv at ROOT_DIR
    index_csv = os.path.join(ROOT_DIR, "index.csv")
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = index_rows[0].keys() if index_rows else [
            "scene_folder", "timestamp", "b04_red_path", "b03_green_path", "b02_blue_path", "out_png", "out_tif"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in index_rows:
            w.writerow(r)

    print(f"Done. Wrote {len(index_rows)} rows to {index_csv}")