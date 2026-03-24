ROOT_NAME = "S2_RGB_no_clouds"

LABEL_PATHS = {
    "2020": "../WorldCover_labels_2020/ESA_WorldCover_10m_2020_v100_N48E009/ESA_WorldCover_10m_2020_v100_N48E009_Map_aligned.tif",
    "2021": "../WorldCover_labels_2021/ESA_WorldCover_10m_2021_v200_N48E009/ESA_WorldCover_10m_2021_v200_N48E009_Map_aligned.tif",
}

TRAIN_FILES = ["urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2B_20200327T101629_32UPV_TOC_V210__20200327T101629",
               "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20200809T102031_32UPV_TOC_V210__20200809T102031",
               "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20200918T102031_32UPV_TOC_V210__20200918T102031"]


# different season - model won't see it in training
TEST_FILES_TEMP = [
    "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2B_20210220T101939_32UPV_TOC_V210__20210220T101939",
    "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20211222T102441_32UPV_TOC_V210__20211222T102441",
]

TARGET_COLS2020 = ["urban_prop_2020", "water_prop_2020", "vegetation_prop_2020"]
TARGET_COLS2021 = ["urban_prop_2021", "water_prop_2021", "vegetation_prop_2021"]

WC_GROUPS = {
    "urban": {50},
    "water": {80, 90, 95},
    "vegetation": {10, 20, 30, 40, 60, 100},
}

FEATURE_COLS_BASELINE = [
    'x', 'y', 'x_norm', 'y_norm',
    'year', 'month', 'day', 'day_of_year', 'doy_sin', 'doy_cos',
    'center_b02', 'mean_b02', 'std_b02', 'min_b02', 'max_b02', 'center_b03',
    'mean_b03', 'std_b03', 'min_b03', 'max_b03', 'center_b04', 'mean_b04',
    'std_b04', 'min_b04', 'max_b04', 'center_b08', 'mean_b08', 'std_b08',
    'min_b08', 'max_b08', 'center_b11', 'mean_b11', 'std_b11', 'min_b11',
    'max_b11'
]

FEATURE_COLS_BASELINE_EXTENDED = ['x', 'y', 'x_norm', 'y_norm',
       'year', 'month', 'day', 'day_of_year', 'doy_sin', 'doy_cos',
       'center_b02', 'mean_b02', 'std_b02', 'min_b02', 'max_b02', 'center_b03',
       'mean_b03', 'std_b03', 'min_b03', 'max_b03', 'center_b04', 'mean_b04',
       'std_b04', 'min_b04', 'max_b04', 'center_b08', 'mean_b08', 'std_b08',
       'min_b08', 'max_b08', 'center_b11', 'mean_b11', 'std_b11', 'min_b11',
       'max_b11', 'mean_NIR', 'mean_SWIR', 'mean_NDVI', 'mean_NDBI',
       'std_NDVI', 'cloud_pct']


FEATURE_COLS_J = ['centroid_x', 'centroid_y', 'area_m2', 'curr_label__built_up',
       'curr_label__vegetation', 'curr_label__water', 'curr_label__other',
       'class_sum', 'B02_median', 'B02_p25', 'B02_p75', 'B02_std',
       'B03_median', 'B03_p25', 'B03_p75', 'B03_std', 'B04_median', 'B04_p25',
       'B04_p75', 'B04_std', 'B08_median', 'B08_p25', 'B08_p75', 'B08_std',
       'B11_median', 'B11_p25', 'B11_p75', 'B11_std', 'B12_median', 'B12_p25',
       'B12_p75', 'B12_std', 'ndvi_median', 'ndvi_p25', 'ndvi_p75', 'ndvi_std',
       'ndwi_median', 'ndwi_p25', 'ndwi_p75', 'ndwi_std', 'ndbi_median',
       'ndbi_p25', 'ndbi_p75', 'ndbi_std', 'swir_used']

TARGET_COLS_TPLUS1_J = ['next_built_up', 'next_vegetation', 'next_water', 'next_other']
TARGET_COLS_DELTA_J = ['delta_built_up', 'delta_vegetation', 'delta_water', 'delta_other']