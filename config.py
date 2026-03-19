ROOT_NAME = "S2_RGB_no_clouds"

# it maps the pictures from 2020 and 2021 into pairs which lies near the similar timeframe
DIRS2USE = {"urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2B_20200327T101629_32UPV_TOC_V210__20200327T101629":
            "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2B_20210302T101839_32UPV_TOC_V210__20210302T101839",
                    
            "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2B_20200426T101549_32UPV_TOC_V210__20200426T101549":
            "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20210426T102021_32UPV_TOC_V210__20210426T102021",
            
            "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20200809T102031_32UPV_TOC_V210__20200809T102031":
            "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20210814T102031_32UPV_TOC_V210__20210814T102031",
            
            "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20200918T102031_32UPV_TOC_V210__20200918T102031":
            "urn:eop:VITO:TERRASCOPE_S2_TOC_V2:S2A_20210923T102031_32UPV_TOC_V210__20210923T102031"}

RGB_STR2LABEL_STR = {"0_102_0": "tree_cover",
                    "204_153_102": "shrubland",
                    "255_255_0": "grassland",
                    "204_0_0": "built_up",
                    "0_0_255": "water_bodies",
                    "0_153_204": "herbaceous_wetland",
                    "153_255_153": "moss_lichen"}

FEATURE_COLS = [
    "x",
    "y",
    "x_norm",
    "y_norm",
    "year",
    "month",
    "day",
    "day_of_year",
    "doy_sin",
    "doy_cos",
    "center_r",
    "center_g",
    "center_b",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "min_r",
    "min_g",
    "min_b",
    "max_r",
    "max_g",
    "max_b",
]

TARGET_COL = "label_id"

GROUPS = {
    "urban": ["built_up"],
    "water": ["water_bodies", "herbaceous_wetland"],
    "vegetation": ["tree_cover", "grassland", "moss_lichen", "shrubland"]
}
