from pathlib import Path
from rasterio.warp import Resampling


WC2020 = Path(
    "WorldCover_labels_2020/ESA_WorldCover_10m_2020_v100_N48E009/ESA_WorldCover_10m_2020_v100_N48E009_Map.tif"
)

WC2021 = Path(
    "WorldCover_labels_2021/ESA_WorldCover_10m_2021_v200_N48E009/ESA_WorldCover_10m_2021_v200_N48E009_Map.tif"
)

SCENES_ROOT = Path("S2_RGB_no_clouds")

# We use WC2021 as the canonical target grid.
MASTER_GRID_PATH = WC2021


# If True: overwrite original files in place.
# If False: write new files with suffixes / new names.
OVERWRITE = False

# Skip output generation if target file already exists.
SKIP_IF_EXISTS = True


WC2020_ALIGNED_NAME = "ESA_WorldCover_10m_2020_v100_N48E009_Map_aligned.tif"
WC2021_ALIGNED_NAME = "ESA_WorldCover_10m_2021_v200_N48E009_Map_aligned.tif"


# Which tif files inside each scene folder should be treated as raw rasters.
# You can narrow this later if needed.
BAND_GLOB = "*.tif"

# Suffix for aligned raw bands when OVERWRITE=False
RAW_SUFFIX = "__aligned_to_wc2021"

# Resampling for continuous satellite bands
RAW_RESAMPLING = Resampling.bilinear

# Resampling for categorical label rasters
LABEL_RESAMPLING = Resampling.nearest



# If not None, only filenames containing one of these tokens will be processed.
# Example:
# BAND_FILENAME_MUST_CONTAIN = ["B02", "B03", "B04", "B08", "B11"]
BAND_FILENAME_MUST_CONTAIN = None

# If True, do not process files that already look like generated outputs
SKIP_DERIVED_FILES = True