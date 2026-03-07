import os
import re
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt


COLOR_MAP = {
    10: (0.0, 0.4, 0.0),
    20: (0.4, 0.8, 0.0),
    30: (0.6, 1.0, 0.6),
    40: (1.0, 1.0, 0.0),
    50: (0.8, 0.0, 0.0),
    60: (0.8, 0.6, 0.4),
    80: (0.0, 0.0, 1.0),
    90: (0.0, 0.6, 0.8),
    95: (0.0, 0.8, 0.6),
    100:(1.0, 1.0, 1.0),
}

def wc_path_for_timestamp(ts: str) -> str:
    year = int(ts[:4])
    if year == 2020:
        return WC2020
    if year == 2021:
        return WC2021
    raise ValueError(f"No WorldCover configured for year={year}")

def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for v, c in COLOR_MAP.items():
        rgb[labels == v] = c
    return rgb

if __name__ == "__main__":
    ROOT_DIR = "S2_RGB_no_clouds"  # scences directories
    WC2020 = "WorldCover_labels_2020/ESA_WorldCover_10m_2020_v100_N48E009/ESA_WorldCover_10m_2020_v100_N48E009_Map.tif"
    WC2021 = "WorldCover_labels_2021/ESA_WorldCover_10m_2021_v200_N48E009/ESA_WorldCover_10m_2021_v200_N48E009_Map.tif"

    # directories timestamps
    TS_RE = re.compile(r"__(\d{8}T\d{6})$")

    scene_dirs = sorted([p for p in glob.glob(os.path.join(ROOT_DIR, "*")) if os.path.isdir(p)])
    print("Scenes:", len(scene_dirs))

    for scene_dir in scene_dirs:
        folder = os.path.basename(scene_dir)
        m = TS_RE.search(folder)
        if not m:
            print("[SKIP] can't parse ts:", folder)
            continue
        ts = m.group(1)

        ref_tif = glob.glob(os.path.join(scene_dir, "*__RGB_cropped.tif"))
        if not ref_tif:
            print("[SKIP] no RGB_cropped.tif in:", folder)
            continue
        ref_tif = ref_tif[0]

        wc_tif = wc_path_for_timestamp(ts)

        # --- open reference RGB crop ---
        with rasterio.open(ref_tif) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_h, ref_w = ref.height, ref.width

            # --- open worldcover ---
            with rasterio.open(wc_tif) as wc:
                src_labels = wc.read(1)

                dst_labels = np.zeros((ref_h, ref_w), dtype=np.uint16)

                reproject(
                    source=src_labels,
                    destination=dst_labels,
                    src_transform=wc.transform,
                    src_crs=wc.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest,  # IMPORTANT for labels
                )

            # save aligned label GeoTIFF
            out_label_tif = os.path.join(scene_dir, f"{folder}__label_aligned.tif")
            meta = ref.meta.copy()
            meta.update({"count": 1, "dtype": "uint16"})
            with rasterio.open(out_label_tif, "w", **meta) as dst:
                dst.write(dst_labels, 1)

        # save colored PNG for quick look (same HxW as RGB crop)
        out_label_png = os.path.join(scene_dir, f"{folder}__label_aligned.png")
        plt.imsave(out_label_png, labels_to_rgb(dst_labels))

        print("[OK]", folder, "->", out_label_tif)