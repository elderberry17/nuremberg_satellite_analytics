from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Config + path helpers
# -----------------------------
CONFIG_PATH = Path("../config/project_config.yaml")  # adjust if needed


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = config_path.resolve().parent
    return cfg


def resolve(cfg: dict, rel: str | Path) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (cfg["_config_dir"] / p).resolve()


cfg = load_config(CONFIG_PATH)

ARTIFACTS_ROOT = resolve(cfg, "../artifacts")
EVAL_DIR = ARTIFACTS_ROOT / "evaluation"
METRICS_DIR = ARTIFACTS_ROOT / "metrics"
PREDS_DIR = ARTIFACTS_ROOT / "predictions"
FIG_DIR = ARTIFACTS_ROOT / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GRID_PATH = resolve(cfg, cfg["paths"]["grid_file"])
SUMMARY_PATH = resolve(cfg, cfg["paths"]["dual_evaluation_summary"])
UNC_PATH = resolve(cfg, cfg["paths"]["uncertainty_table"])

# Set your preferred models here for the more detailed plots
DEFAULT_TASK = "delta"          # "delta" or "tplus1"
DEFAULT_MODEL = "elastic_net"   # e.g. "elastic_net", "random_forest", "gradient_boosting"


# -----------------------------
# Style
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def model_display_name(name: str) -> str:
    return {
        "elastic_net": "Elastic Net",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }.get(name, name.replace("_", " ").title())


def task_display_name(name: str) -> str:
    return {"delta": "Delta", "tplus1": "T+1"}.get(name, name)


# -----------------------------
# 1) Macro MAE bar chart
# -----------------------------
def plot_macro_mae(summary: pd.DataFrame) -> None:
    df = summary.copy()

    # Native-task metric only
    df["native_macro_mae"] = np.where(
        df["task"] == "delta",
        df["macro_mae_delta"],
        df["macro_mae_next_year"],
    )

    tasks = ["delta", "tplus1"]
    models = ["elastic_net", "random_forest", "gradient_boosting"]

    x = np.arange(len(tasks))
    width = 0.23

    fig, ax = plt.subplots(figsize=(6.8, 3.6))

    for i, m in enumerate(models):
        vals = []
        for t in tasks:
            row = df[(df["task"] == t) & (df["model"] == m)]
            vals.append(float(row["native_macro_mae"].iloc[0]) if not row.empty else np.nan)

        ax.bar(x + (i - 1) * width, vals, width=width, label=model_display_name(m))

    ax.set_xticks(x)
    ax.set_xticklabels([task_display_name(t) for t in tasks])
    ax.set_ylabel("Macro MAE")
    ax.set_title("Macro MAE on Each Model’s Native Task")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_macro_mae_by_task.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 2) Change-quality metrics
# -----------------------------
def plot_change_quality(summary: pd.DataFrame) -> None:
    df = summary.copy()
    df["label"] = df["task"].map(task_display_name) + " - " + df["model"].map(model_display_name)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    order = list(df.sort_values(["task", "model"])["label"])

    left = df.set_index("label").loc[order]
    axes[0].barh(left.index, left["false_change_rate"])
    axes[0].set_title("False Change Rate")
    axes[0].set_xlabel("Rate")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(left.index, left["stability_score"])
    axes[1].set_title("Stability Score")
    axes[1].set_xlabel("Score")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("Change-Aware Evaluation Metrics", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_change_quality_metrics.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 3) Stress-test curves
# -----------------------------
def plot_stress_tests() -> None:
    stress_files = sorted(EVAL_DIR.glob("*_stress_test.csv"))
    if not stress_files:
        print("No stress-test CSVs found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)

    for task_name, ax in zip(["delta", "tplus1"], axes):
        task_files = [p for p in stress_files if p.name.startswith(f"{task_name}_")]
        for p in task_files:
            df = pd.read_csv(p)
            if "macro_mae_delta" in df.columns:
                ycol = "macro_mae_delta"
            else:
                ycol = "macro_mae_next_year"

            model = p.name.replace(f"{task_name}_", "").replace("_stress_test.csv", "")
            ax.plot(
                df["fraction"] * 100,
                df[ycol],
                marker="o",
                linewidth=1.8,
                label=model_display_name(model)
            )

        ax.set_title(f"{task_display_name(task_name)} Stress Test")
        ax.set_xlabel("Randomly Masked Input Values (%)")
        ax.set_ylabel("Macro MAE")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle("Robustness Under Missingness", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_stress_test_curves.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 4) Per-target MAE heatmap
# -----------------------------
def plot_per_target_mae_heatmap() -> None:
    metric_files = sorted(METRICS_DIR.glob("*_metrics.csv"))
    if not metric_files:
        print("No metrics CSVs found.")
        return

    rows = []
    for p in metric_files:
        df = pd.read_csv(p)

        # Expect filenames like:
        # delta_elastic_net_metrics.csv
        # tplus1_random_forest_metrics.csv
        stem = p.stem.replace("_metrics", "")  # e.g. "delta_elastic_net"

        if stem.startswith("delta_"):
            task = "delta"
            model = stem[len("delta_"):]
        elif stem.startswith("tplus1_"):
            task = "tplus1"
            model = stem[len("tplus1_"):]
        else:
            print(f"Skipping unrecognized metrics filename: {p.name}")
            continue

        # Basic column checks
        required_cols = {"target", "mae"}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {p.name}: missing required columns {required_cols - set(df.columns)}")
            continue

        df = df.copy()
        df["task"] = task
        df["model"] = model

        # remove macro row for heatmap
        df = df[df["target"] != "macro_avg"].copy()

        rows.append(df[["task", "model", "target", "mae"]])

    if not rows:
        print("No valid per-target metrics found.")
        return

    all_metrics = pd.concat(rows, ignore_index=True)
    all_metrics["row_label"] = (
        all_metrics["task"].map(task_display_name)
        + " - "
        + all_metrics["model"].map(model_display_name)
    )

    pivot = all_metrics.pivot(index="row_label", columns="target", values="mae")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.55 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Per-Target MAE Across Task/Model Pairs")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MAE")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_per_target_mae_heatmap.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 5) Residual boxplots for one model-task pair
# -----------------------------
def plot_residual_boxplots(task_name: str = DEFAULT_TASK, model_name: str = DEFAULT_MODEL) -> None:
    pred_path = PREDS_DIR / f"{task_name}_{model_name}_test_predictions.parquet"
    if not pred_path.exists():
        print(f"Prediction parquet not found: {pred_path}")
        return

    df = pd.read_parquet(pred_path)

    resid_cols = [c for c in df.columns if c.startswith("resid_")]
    if not resid_cols:
        print("No residual columns found.")
        return

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    data = [df[c].dropna().values for c in resid_cols]
    ax.boxplot(data, tick_labels=[c.replace("resid_", "") for c in resid_cols], showfliers=False)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(f"Residual Distributions: {task_display_name(task_name)} - {model_display_name(model_name)}")
    ax.set_ylabel("Residual (true - predicted)")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_residual_boxplots.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 6) Uncertainty histograms
# -----------------------------
def plot_uncertainty_histograms() -> None:
    if not UNC_PATH.exists():
        print(f"Uncertainty file not found: {UNC_PATH}")
        return

    unc = pd.read_parquet(UNC_PATH)
    std_cols = [c for c in unc.columns if "_uncertainty_std_" in c]

    if not std_cols:
        print("No uncertainty std columns found.")
        return

    n = len(std_cols)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, std_cols):
        vals = pd.to_numeric(unc[col], errors="coerce").dropna()
        ax.hist(vals, bins=25)
        ax.set_title(col.replace("_uncertainty_std_", "\nstd: "))
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)

    for ax in axes[len(std_cols):]:
        ax.axis("off")

    fig.suptitle("Random-Forest Uncertainty Distributions", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_uncertainty_histograms.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 7) Spatial error map for one target
# -----------------------------
def plot_spatial_error_map(task_name: str = DEFAULT_TASK, model_name: str = DEFAULT_MODEL) -> None:
    pred_path = PREDS_DIR / f"{task_name}_{model_name}_test_predictions.parquet"
    if not pred_path.exists():
        print(f"Prediction parquet not found: {pred_path}")
        return

    grid = gpd.read_file(GRID_PATH)
    preds = pd.read_parquet(pred_path)

    # choose a meaningful default target
    target_candidates = [c for c in preds.columns if c.startswith("resid_delta_built_up") or c == "resid_delta_built_up"]
    if target_candidates:
        resid_col = target_candidates[0]
    else:
        resid_cols = [c for c in preds.columns if c.startswith("resid_")]
        if not resid_cols:
            print("No residual columns found for spatial map.")
            return
        resid_col = resid_cols[0]

    merged = grid.merge(preds[["grid_id", resid_col]], on="grid_id", how="inner")

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    merged.plot(column=resid_col, ax=ax, legend=True)
    ax.set_title(
        f"Spatial Residual Map\n{task_display_name(task_name)} - {model_display_name(model_name)}\n{resid_col}"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_spatial_error_map.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Optional 8) Predicted vs true scatter for native targets
# -----------------------------
def plot_true_vs_pred_scatter(task_name: str = DEFAULT_TASK, model_name: str = DEFAULT_MODEL) -> None:
    pred_path = PREDS_DIR / f"{task_name}_{model_name}_test_predictions.parquet"
    if not pred_path.exists():
        print(f"Prediction parquet not found: {pred_path}")
        return

    df = pd.read_parquet(pred_path)

    if task_name == "delta":
        true_cols = [c for c in df.columns if c.startswith("delta_")]
        pred_cols = [f"pred_{c}" for c in true_cols if f"pred_{c}" in df.columns]
    else:
        true_cols = [c for c in df.columns if c.startswith("next_")]
        pred_cols = [f"pred_{c}" for c in true_cols if f"pred_{c}" in df.columns]

    if not true_cols or not pred_cols:
        print("No matched true/pred columns found for scatter plot.")
        return

    n = len(true_cols)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(8.8, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, true_col in zip(axes, true_cols):
        pred_col = f"pred_{true_col}"
        if pred_col not in df.columns:
            ax.axis("off")
            continue

        x = pd.to_numeric(df[true_col], errors="coerce")
        y = pd.to_numeric(df[pred_col], errors="coerce")
        valid = x.notna() & y.notna()

        ax.scatter(x[valid], y[valid], s=10, alpha=0.6)
        lo = min(x[valid].min(), y[valid].min())
        hi = max(x[valid].max(), y[valid].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_title(true_col)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.25)

    for ax in axes[len(true_cols):]:
        ax.axis("off")

    fig.suptitle(f"True vs Predicted: {task_display_name(task_name)} - {model_display_name(model_name)}", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_true_vs_pred_scatter.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_PATH}")

    summary = pd.read_csv(SUMMARY_PATH)

    plot_macro_mae(summary)
    plot_change_quality(summary)
    plot_stress_tests()
    plot_per_target_mae_heatmap()
    plot_residual_boxplots(task_name=DEFAULT_TASK, model_name=DEFAULT_MODEL)
    plot_uncertainty_histograms()
    plot_spatial_error_map(task_name=DEFAULT_TASK, model_name=DEFAULT_MODEL)
    plot_true_vs_pred_scatter(task_name=DEFAULT_TASK, model_name=DEFAULT_MODEL)

    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()