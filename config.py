ROOT_NAME = "S2_RGB_no_clouds"

LABELS_PATH = "WorldCover_labels_2021/ESA_WorldCover_10m_2021_v200_N48E009/ESA_WorldCover_10m_2021_v200_N48E009_Map_aligned.tif"

TRAIN_FILES = ["urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2B_20200327T101629_32UPV_TOC_V210__20200327T101629",
            #    "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2B_20200416T101549_32UPV_TOC_V210__20200416T101549", 
               "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20200809T102031_32UPV_TOC_V210__20200809T102031",
               "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20200918T102031_32UPV_TOC_V210__20200918T102031"]


TEST_FILES_TEMP = ["urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20200713T103031_32UPV_TOC_V210__20200713T103031",
                   "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2B_20200406T101559_32UPV_TOC_V210__20200406T101559"]


WC_GROUPS = {
    "urban": {50},
    "water": {80, 90, 95},
    "vegetation": {10, 20, 30, 40, 60, 100},
}

FEATURE_COLS = [
    'x', 'y', 'x_norm', 'y_norm',
    'year', 'month', 'day', 'day_of_year', 'doy_sin', 'doy_cos',
    'center_b02', 'mean_b02', 'std_b02', 'min_b02', 'max_b02', 'center_b03',
    'mean_b03', 'std_b03', 'min_b03', 'max_b03', 'center_b04', 'mean_b04',
    'std_b04', 'min_b04', 'max_b04', 'center_b08', 'mean_b08', 'std_b08',
    'min_b08', 'max_b08', 'center_b11', 'mean_b11', 'std_b11', 'min_b11',
    'max_b11'
]