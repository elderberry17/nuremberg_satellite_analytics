from models.inference_utils import (
    load_parquet,
    load_model,
    run_inference_and_format,
    save_predictions_parquet,
)

from config import FEATURE_COLS_J


def main():
    df = load_parquet("./datasets/df_to_inference.parquet")
    model = load_model(
        "./results_repro/exp_2026-03-23_21-01-14-change/mlp/change_mlp_extended_spatial_2026-03-23_21-15-23.pkl"
    )
    out_df = run_inference_and_format(model, df, FEATURE_COLS_J, "change")
    save_predictions_parquet(out_df, "./inference_outputs/change.parquet")


if __name__ == "__main__":
    main()
