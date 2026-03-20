"""
Download configuration for `process_esa/*_download.py` scripts.

Edit values here to download only the time window / products you need.
"""

# -----------------------
# Sentinel-2 raw RGB download
# -----------------------

RAW_OUT_DIR = "S2_RGB_no_clouds"

# 20210220T101939
# 20211222T102441

# Geographic bounds used for searching products (minx, miny, maxx, maxy)
RAW_BOUNDS = (10.95, 49.38, 11.15, 49.52)

RAW_PRODUCT_URN = "urn:eop:VITO:TERRASCOPE_S2_TOC_V2"

# Time window for raw scene download (ISO dates)
RAW_START = "2021-12-22"
RAW_END = "2021-12-23"

# Terracatalogue query options
RAW_CLOUD_COVER = 5
RAW_LIMIT = 200

# Which band files to download (Blue, Green, Red at 10m)
RAW_KEEP_TOKENS = [
    "TOC-B02_10M",
    "TOC-B03_10M",
    "TOC-B04_10M",
    "TOC-B08_10M",
    "TOC-B11_20M"
]

# -----------------------
# WorldCover label download
# -----------------------

LABEL_BOUNDS = RAW_BOUNDS

# WorldCover product specs per year
WORLD_COVER_SPECS = {
    2020: {
        "urn": "urn:eop:VITO:ESA_WorldCover_10m_2020_V1",
        "out_dir": "WorldCover_labels_2020",
    },
    2021: {
        "urn": "urn:eop:VITO:ESA_WorldCover_10m_2021_V2",
        "out_dir": "WorldCover_labels_2021",
    },
}

# Which label years to download (edit as needed)
LABEL_YEARS = [2020, 2021]