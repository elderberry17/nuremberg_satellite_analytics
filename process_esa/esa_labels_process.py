import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
from shapely.geometry import Polygon, mapping
import pyproj
from shapely.ops import transform as shp_transform

if __name__ == "__main__":
    tif_path = "WorldCover_labels_2021/ESA_WorldCover_10m_2021_v200_N48E009/ESA_WorldCover_10m_2021_v200_N48E009_Map.tif"

    roi_coords = [
        (10.95, 49.52),
        (11.15, 49.52),
        (11.15, 49.38),
        (10.95, 49.38),
        (10.95, 49.52),
    ]
    roi_poly_wgs84 = Polygon(roi_coords)

    color_map = {
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

    with rasterio.open(tif_path) as src:
        # ROI -> CRS растра (на всякий случай)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        roi_proj = shp_transform(transformer.transform, roi_poly_wgs84)

        arr, _ = mask(src, [mapping(roi_proj)], crop=True)
        labels = arr[0]

    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    for v, c in color_map.items():
        rgb[labels == v] = c

    plt.imsave("WorldCover_labels_2021/ESA_WorldCover_10m_2021_v200_N48E009/WorldCover_2021_colored_ROI.png", rgb)