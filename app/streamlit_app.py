from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import yaml
import plotly.graph_objects as go
import plotly.express as px


# CartoDB GL styles + Mapbox satellite
MAPBOX_TOKEN = "TOKEN_GOES_HERE"

BASEMAPS = {
    "Dark Matter":    "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "Light":          "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "Voyager":        "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    "Dark No Labels": "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json",
    "Light No Labels":"https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json",
    "Satellite":      "mapbox://styles/mapbox/satellite-streets-v12",
}

# project config lives next to this script
CONFIG_PATH = Path(__file__).resolve().with_name("project_config.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = str(CONFIG_PATH.resolve().parent)
    return cfg


def resolve_path(cfg, rel):
    p = Path(rel)
    if p.is_absolute():
        return p
    return (Path(cfg["_config_dir"]) / p).resolve()


cfg = load_config()
PRED_PATH   = resolve_path(cfg, cfg["paths"]["app_predictions"])
EVAL_PATH   = resolve_path(cfg, cfg["paths"]["dual_evaluation_summary"])
UNC_PATH    = resolve_path(cfg, cfg["paths"]["uncertainty_table"])
CELL_SIZE_M = int(cfg["grid"]["cell_size_m"])


st.set_page_config(page_title="NürnbergLens", page_icon="🛰️", layout="wide")

# a bit of custom styling — metric cards, section headers, pills
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; color: #f0f0f0; }
[data-testid="stMetricLabel"] { font-size: 0.72rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; }
.section-hdr {
    font-size: 1rem;
    font-weight: 700;
    color: #e8593c;
    border-left: 3px solid #e8593c;
    padding-left: 10px;
    margin: 1.2rem 0 0.6rem;
}
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.pill-red   { background: #e8593c22; color: #e8593c; border: 1px solid #e8593c55; }
.pill-green { background: #2d9e5f22; color: #2d9e5f; border: 1px solid #2d9e5f55; }
.pill-blue  { background: #3b82f622; color: #3b82f6; border: 1px solid #3b82f655; }
.pill-amber { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b55; }
</style>
""", unsafe_allow_html=True)


# ---- data loading ----

@st.cache_data
def load_predictions(path):
    if not path.exists():
        return None
    return gpd.read_file(path)


@st.cache_data
def load_eval_summary(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_uncertainty_table(path):
    if not path.exists():
        return None
    return pd.read_parquet(path)


gdf     = load_predictions(PRED_PATH)
eval_df = load_eval_summary(EVAL_PATH)
unc_df  = load_uncertainty_table(UNC_PATH)

if gdf is None:
    st.error("app_predictions.geojson not found. Run the prediction generation script first.")
    st.stop()

if gdf.crs is None:
    st.error("The prediction GeoJSON has no CRS — something went wrong during export.")
    st.stop()


# ---- helper functions ----

def pretty_label(col):
    return col.replace("_", " ").title()


def get_uncertainty_cols(df):
    return [c for c in df.columns if "uncertainty_" in c.lower() or c.endswith("_std")]


def build_metric_catalog(df):
    # figure out which observed years are in the data
    observed_years = sorted({
        int(c.split("_")[1]) for c in df.columns
        if c.startswith("observed_") and len(c.split("_")) >= 3
    })
    forecast_year = int(df["forecast_year"].iloc[0]) if "forecast_year" in df.columns else 2022

    catalog = {}
    for yr in observed_years:
        catalog[yr] = {
            "Observed composition": [c for c in df.columns if c.startswith(f"observed_{yr}_")]
        }

    catalog.setdefault(forecast_year, {})
    catalog[forecast_year]["Forecast composition"] = [
        c for c in df.columns
        if c.startswith(f"forecast_{forecast_year}_")
        or c.startswith(f"ensemble_{forecast_year}_")
        or c.startswith(f"direct_reconstructed_{forecast_year}_")
    ]
    catalog[forecast_year]["Predicted change"] = [
        c for c in df.columns
        if c.startswith(f"derived_delta_{forecast_year}_")
        or c.startswith(f"direct_delta_{forecast_year}_")
        or c.startswith(f"ensemble_delta_{forecast_year}_")
    ]

    # drop empty groups
    return {yr: {g: cols for g, cols in groups.items() if cols} for yr, groups in catalog.items()}


def pick_default_metric(group_name, metrics):
    preferred = ["ensemble_delta_2022_built_up", "forecast_2022_built_up", "observed_2021_built_up"]
    for p in preferred:
        if p in metrics:
            return p
    return metrics[0]


def safe_mean(df, col):
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").mean())


# ---- color helpers ----
# each function takes a pandas Series and returns a list of [R,G,B,A] lists

def color_signed(values):
    max_abs = max(float(np.nanmax(np.abs(values))), 1e-6)
    result = []
    for v in values:
        if pd.isna(v):
            result.append([150, 150, 150, 60])
            continue
        t = min(1.0, abs(float(v)) / max_abs)
        alpha = int(40 + 215 * t)
        result.append([220, 70, 70, alpha] if v >= 0 else [60, 110, 230, alpha])
    return result, max_abs


def color_magnitude(values):
    max_val = max(float(np.nanmax(values)), 1e-6)
    result = []
    for v in values:
        if pd.isna(v):
            result.append([150, 150, 150, 60])
            continue
        t = min(1.0, float(v) / max_val)
        alpha = int(40 + 215 * t)
        result.append([int(220 + 35 * t), int(80 + 160 * t), 20, alpha])
    return result, max_val


def color_vegetation(values):
    max_val = max(float(np.nanmax(values)), 1e-6)
    result = []
    for v in values:
        if pd.isna(v):
            result.append([150, 150, 150, 60])
            continue
        t = min(1.0, float(v) / max_val)
        alpha = int(50 + 200 * t)
        result.append([int(20 + 40 * (1 - t)), int(100 + 155 * t), int(20 + 30 * (1 - t)), alpha])
    return result, max_val


def color_water(values):
    max_val = max(float(np.nanmax(values)), 1e-6)
    result = []
    for v in values:
        if pd.isna(v):
            result.append([150, 150, 150, 60])
            continue
        t = min(1.0, float(v) / max_val)
        alpha = int(50 + 200 * t)
        result.append([30, int(80 + 120 * t), int(180 + 75 * t), alpha])
    return result, max_val


def rgb_composition(row, year="2021"):
    # blend red/green/blue from urban/veg/water proportions
    urban = float(row.get(f"observed_{year}_built_up", 0) or 0)
    veg   = float(row.get(f"observed_{year}_vegetation", 0) or 0)
    water = float(row.get(f"observed_{year}_water", 0) or 0)
    r = int(min(255, urban * 255 * 2.5))
    g = int(min(255, veg   * 255 * 1.2))
    b = int(min(255, water * 255 * 4.0))
    return [r, g, b, 200]


def pick_color_fn(metric, color_mode):
    if color_mode == "Signed value":
        return color_signed
    if "vegetation" in metric:
        return color_vegetation
    if "water" in metric:
        return color_water
    return color_magnitude


def build_map_layers(gdf, metric, color_mode, show_unc, unc_col, use_composition=False, comp_year="2021"):
    polygons = gdf.to_crs("EPSG:4326").copy()
    polygons[metric] = pd.to_numeric(polygons[metric], errors="coerce").fillna(0)

    if use_composition:
        polygons["color"] = polygons.apply(lambda row: rgb_composition(row, comp_year), axis=1)
        scale = "RGB composition"
    else:
        color_fn = pick_color_fn(metric, color_mode)
        polygons["color"], scale = color_fn(polygons[metric])

    # compute centroids for scatter/uncertainty layer
    centroids = polygons.copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids["lon"] = centroids.geometry.x
    centroids["lat"] = centroids.geometry.y
    centroids["radius"] = 140

    active_unc = None
    if show_unc and unc_col != "None" and unc_col in centroids.columns:
        unc_vals = pd.to_numeric(centroids[unc_col], errors="coerce").fillna(0)
        centroids["radius"] = 60 + 320 * unc_vals / max(float(unc_vals.max()), 1e-6)
        active_unc = unc_col

    legend_text = "RGB composition" if use_composition else f"max |value| = {scale:.4f}"
    return polygons, centroids, polygons.__geo_interface__, legend_text, active_unc


def build_hotspot_table(df, metric, top_n, unc_col, rank_by_abs):
    out = df.copy()
    out["score"] = out[metric].abs() if rank_by_abs else out[metric]
    cols = ["grid_id", metric, "score"]
    for extra in ["ensemble_abs_total_change", "ensemble_change_flag", "ensemble_dominant_change_class"]:
        if extra in out.columns:
            cols.append(extra)
    if unc_col and unc_col in out.columns:
        cols.append(unc_col)
    return out.sort_values("score", ascending=False)[cols].head(top_n).reset_index(drop=True)


# ---- sidebar ----

catalog          = build_metric_catalog(gdf)
uncertainty_cols = get_uncertainty_cols(gdf)
available_years  = sorted(catalog.keys())

with st.sidebar:
    st.markdown("## 🛰️ NürnbergLens")
    st.markdown("*Urban Change Intelligence*")
    st.markdown("---")

    st.markdown("### 🗓️ Time & Layer")
    selected_year = st.select_slider("Year", options=available_years, value=available_years[-1])
    group_names   = list(catalog[selected_year].keys())
    metric_group  = st.selectbox("View family", group_names)
    sel_metrics   = catalog[selected_year][metric_group]
    metric        = st.selectbox(
        "Layer",
        sel_metrics,
        index=sel_metrics.index(pick_default_metric(metric_group, sel_metrics)),
        format_func=pretty_label,
    )

    st.markdown("---")
    st.markdown("### 🗺️ Map")
    basemap_name    = st.selectbox("Basemap", list(BASEMAPS.keys()))
    use_composition = st.checkbox("🎨 RGB view (green=veg, red=urban, blue=water)", value=True)
    comp_year       = st.radio("Composition year", ["2020", "2021"], horizontal=True) if use_composition else "2021"
    color_mode      = st.radio("Color mode", ["Signed value", "Absolute magnitude"])
    opacity         = st.slider("Opacity", 0.1, 1.0, 0.65, 0.05)

    st.markdown("---")
    st.markdown("### 🔍 Uncertainty")
    show_unc = st.checkbox("Show uncertainty rings", value=bool(uncertainty_cols))
    unc_col  = st.selectbox("Uncertainty field", ["None"] + uncertainty_cols) if show_unc else "None"

    st.markdown("---")
    st.markdown("### 📊 Hotspots")
    top_n    = st.slider("Rows", 10, 100, int(cfg["reporting"]["hotspot_export_rows"]), 5)
    abs_rank = st.checkbox("Rank by absolute value", value=True)

    st.markdown("---")
    st.warning(f"""**⚠️ Limitations**
- {CELL_SIZE_M}m grid cells only
- ESA WorldCover label noise
- 2020→2021 transition only
- Not for legal decisions""")


# ---- prepare data ----

grid_view, centroids, poly_json, legend, active_unc = build_map_layers(
    gdf, metric, color_mode, show_unc, unc_col, use_composition, comp_year
)


# ---- page header ----

st.title("🛰️ NürnbergLens — Urban Change Intelligence")
st.caption("Interactive screening tool for land cover, change predictions, and t+1 forecasting to 2022.")


# ---- top KPI row ----

vals = pd.to_numeric(centroids[metric], errors="coerce").dropna()
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Layer",  pretty_label(metric)[:16])
k2.metric("Mean",   f"{vals.mean():.4f}")
k3.metric("Median", f"{vals.median():.4f}")
k4.metric("Min",    f"{vals.min():.4f}")
k5.metric("Max",    f"{vals.max():.4f}")
k6.metric("Cells",  f"{len(centroids):,}")
st.caption(f"**Scale:** {legend}")


# ---- map ----

grid_layer = pdk.Layer(
    "GeoJsonLayer",
    data=poly_json,
    pickable=True,
    stroked=True,
    filled=True,
    get_fill_color="properties.color",
    get_line_color=[30, 30, 30, 120],
    line_width_min_pixels=1,
    opacity=opacity,
)

unc_layer = None
if active_unc:
    unc_layer = pdk.Layer(
        "ScatterplotLayer",
        data=centroids,
        get_position="[lon, lat]",
        get_fill_color=[255, 255, 255, 15],
        get_line_color=[255, 140, 0, 200],
        get_radius="radius",
        stroked=True,
        filled=True,
        line_width_min_pixels=1.5,
        pickable=True,
        opacity=0.5,
    )

view_state = pdk.ViewState(
    latitude=float(centroids["lat"].mean()),
    longitude=float(centroids["lon"].mean()),
    zoom=10.8,
    pitch=0,
)

# rich hover tooltip showing all key values per cell
tooltip_html = """
<div style='background:#1a1a2e;color:#f0f0f0;padding:12px 14px;
            border-radius:10px;font-family:monospace;font-size:12px;
            min-width:260px;border:1px solid #3a3a5a'>
  <b style='font-size:13px;color:#e8593c'>📍 Grid Cell {grid_id}</b>
  <hr style='border-color:#2a2a4a;margin:6px 0'>
  <b style='color:#aaa'>2020 observed</b><br>
  🏙️ Built-up: <b>{observed_2020_built_up}</b> &nbsp;
  🌿 Veg: <b>{observed_2020_vegetation}</b> &nbsp;
  💧 Water: <b>{observed_2020_water}</b><br>
  <b style='color:#aaa'>2021 observed</b><br>
  🏙️ Built-up: <b>{observed_2021_built_up}</b> &nbsp;
  🌿 Veg: <b>{observed_2021_vegetation}</b> &nbsp;
  💧 Water: <b>{observed_2021_water}</b><br>
  <hr style='border-color:#2a2a4a;margin:6px 0'>
  <b style='color:#aaa'>2022 forecast</b><br>
  🏙️ Built-up: <b>{forecast_2022_built_up}</b> &nbsp;
  🌿 Veg: <b>{forecast_2022_vegetation}</b><br>
  <b style='color:#aaa'>Ensemble change</b><br>
  🏙️ Δ Built-up: <b>{ensemble_delta_2022_built_up}</b> &nbsp;
  🌿 Δ Veg: <b>{ensemble_delta_2022_vegetation}</b><br>
  <hr style='border-color:#2a2a4a;margin:6px 0'>
  📊 Total change: <b>{ensemble_abs_total_change}</b> &nbsp;
  🚩 Flag: <b>{ensemble_change_flag}</b><br>
  🔑 Dominant: <b>{ensemble_dominant_change_class}</b>
</div>
"""

layers = [grid_layer]
if unc_layer:
    layers.append(unc_layer)

st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=BASEMAPS[basemap_name],
        map_provider="mapbox",
        api_keys={"mapbox": MAPBOX_TOKEN},
        tooltip={"html": tooltip_html},
    ),
    use_container_width=True,
    height=520,
    key=f"main_map_{basemap_name}",
)

# ---- tabs ----

st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Analytics", "🔥 Hotspots", "📈 Distribution",
    "ℹ️ Interpretation", "🔍 Area Search", "🤖 Model Performance",
])


# ---- tab 1: analytics ----

with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-hdr">🏙️ Urban vs Vegetation vs Water — 3 year view</div>', unsafe_allow_html=True)

        needed = ["observed_2020_built_up", "observed_2021_built_up",
                  "observed_2020_vegetation", "observed_2021_vegetation",
                  "observed_2020_water", "observed_2021_water"]

        if all(c in centroids.columns for c in needed):
            cats  = ["Built-up", "Vegetation", "Water"]
            y2020 = [centroids["observed_2020_built_up"].mean(),
                     centroids["observed_2020_vegetation"].mean(),
                     centroids["observed_2020_water"].mean()]
            y2021 = [centroids["observed_2021_built_up"].mean(),
                     centroids["observed_2021_vegetation"].mean(),
                     centroids["observed_2021_water"].mean()]
            y2022 = [safe_mean(centroids, "forecast_2022_built_up"),
                     safe_mean(centroids, "forecast_2022_vegetation"),
                     safe_mean(centroids, "forecast_2022_water")]

            fig = go.Figure()
            fig.add_trace(go.Bar(name="2020", x=cats, y=y2020,
                                 marker_color=["#e8593c", "#2d9e5f", "#3b82f6"], opacity=0.4))
            fig.add_trace(go.Bar(name="2021", x=cats, y=y2021,
                                 marker_color=["#e8593c", "#2d9e5f", "#3b82f6"], opacity=1.0))
            fig.add_trace(go.Bar(name="2022 forecast", x=cats, y=y2022,
                                 marker_color=["#ff8c69", "#5dbb8c", "#60a5fa"],
                                 opacity=0.8, marker_pattern_shape="/"))
            fig.update_layout(barmode="group", template="plotly_dark",
                               height=300, margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Observed columns not found in the data.")

    with col_right:
        st.markdown('<div class="section-hdr">📉 Change direction — all classes</div>', unsafe_allow_html=True)

        delta_cols = [c for c in centroids.columns if "ensemble_delta" in c][:4]
        if delta_cols:
            labels, inc_v, stab_v, dec_v = [], [], [], []
            for col in delta_cols:
                v = pd.to_numeric(centroids[col], errors="coerce").dropna()
                labels.append(col.replace("ensemble_delta_2022_", "").title())
                inc_v.append(int((v > 0.02).sum()))
                stab_v.append(int((v.abs() <= 0.02).sum()))
                dec_v.append(int((v < -0.02).sum()))

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="↑ Increasing", x=labels, y=inc_v,  marker_color="#e8593c"))
            fig2.add_trace(go.Bar(name="→ Stable",     x=labels, y=stab_v, marker_color="#555577"))
            fig2.add_trace(go.Bar(name="↓ Decreasing", x=labels, y=dec_v,  marker_color="#3b82f6"))
            fig2.update_layout(barmode="stack", template="plotly_dark",
                                height=300, margin=dict(l=0, r=0, t=10, b=0),
                                legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig2, use_container_width=True)

    # city-wide stability summary
    st.markdown('<div class="section-hdr">🌡️ City-wide change metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)

    if "ensemble_abs_total_change" in centroids.columns:
        stability = 1 - pd.to_numeric(centroids["ensemble_abs_total_change"], errors="coerce").fillna(0).clip(0, 1)
        m1.metric("🟢 Highly stable",   f"{(stability > 0.9).sum():,}")
        m2.metric("🟡 Moderate change", f"{((stability >= 0.7) & (stability <= 0.9)).sum():,}")
        m3.metric("🔴 High change",     f"{(stability < 0.7).sum():,}")

    if "ensemble_change_flag" in centroids.columns:
        flagged = pd.to_numeric(centroids["ensemble_change_flag"], errors="coerce").fillna(0)
        m4.metric("🚩 Flagged cells", f"{int(flagged.sum()):,}")

        if "ensemble_abs_total_change" in centroids.columns:
            real_change = (
                pd.to_numeric(centroids["ensemble_abs_total_change"], errors="coerce").fillna(0)
                > float(cfg["modeling"]["change_threshold"])
            ).astype(int)
            false_change  = int(((flagged == 1) & (real_change == 0)).sum())
            total_flagged = max(int(flagged.sum()), 1)
            m5.metric("❌ False change rate", f"{false_change / total_flagged * 100:.1f}%")

    # top 10 cells by urban pressure (current built-up + predicted growth)
    st.markdown('<div class="section-hdr">🏗️ Urban pressure index — top 10 cells</div>', unsafe_allow_html=True)
    if all(c in centroids.columns for c in ["observed_2021_built_up", "ensemble_delta_2022_built_up"]):
        centroids["urban_pressure"] = (
            centroids["observed_2021_built_up"] * 0.5
            + pd.to_numeric(centroids["ensemble_delta_2022_built_up"], errors="coerce").fillna(0).clip(0) * 0.5
        )
        top10 = centroids.nlargest(10, "urban_pressure")[
            ["grid_id", "urban_pressure", "observed_2021_built_up", "ensemble_delta_2022_built_up"]
        ]
        fig3 = px.bar(top10, x="grid_id", y="urban_pressure",
                      color="urban_pressure", color_continuous_scale="Reds",
                      template="plotly_dark", height=250)
        fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)


# ---- tab 2: hotspots ----

with tab2:
    st.markdown('<div class="section-hdr">🔥 Top changing cells</div>', unsafe_allow_html=True)
    hot = build_hotspot_table(centroids, metric, top_n, active_unc, abs_rank)

    def highlight_val(val):
        try:
            v = float(val)
            if v > 0.1:  return "background-color:#e8593c33;color:#e8593c"
            if v < -0.1: return "background-color:#3b82f633;color:#3b82f6"
        except:
            pass
        return ""

    st.dataframe(hot.style.applymap(highlight_val, subset=[metric]),
                 use_container_width=True, height=420)

    btn_col, info_col = st.columns([1, 3])
    with btn_col:
        st.download_button("⬇️ Download CSV",
                           data=hot.to_csv(index=False).encode("utf-8"),
                           file_name="hotspots.csv", mime="text/csv")
    with info_col:
        if "ensemble_change_flag" in hot.columns:
            n_flagged = int(pd.to_numeric(hot["ensemble_change_flag"], errors="coerce").fillna(0).sum())
            st.markdown(f'<span class="pill pill-red">🚩 {n_flagged} flagged in top {top_n}</span>',
                        unsafe_allow_html=True)
        if "ensemble_dominant_change_class" in hot.columns:
            dominant = hot["ensemble_dominant_change_class"].value_counts().idxmax()
            st.markdown(f'<span class="pill pill-amber">🔑 Dominant class: {dominant}</span>',
                        unsafe_allow_html=True)


# ---- tab 3: distribution ----

with tab3:
    st.markdown('<div class="section-hdr">📈 Value distribution — selected layer</div>', unsafe_allow_html=True)

    hist_vals = pd.to_numeric(centroids[metric], errors="coerce").dropna()
    if len(hist_vals):
        fig_hist = px.histogram(hist_vals, nbins=50,
                                color_discrete_sequence=["#e8593c"],
                                template="plotly_dark",
                                labels={"value": pretty_label(metric)})
        fig_hist.add_vline(x=float(hist_vals.mean()),   line_dash="dash", line_color="#f59e0b", annotation_text="mean")
        fig_hist.add_vline(x=float(hist_vals.median()), line_dash="dot",  line_color="#3b82f6", annotation_text="median")
        fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("p10", f"{float(hist_vals.quantile(0.10)):.4f}")
        p2.metric("p25", f"{float(hist_vals.quantile(0.25)):.4f}")
        p3.metric("p50", f"{float(hist_vals.quantile(0.50)):.4f}")
        p4.metric("p75", f"{float(hist_vals.quantile(0.75)):.4f}")
        p5.metric("p90", f"{float(hist_vals.quantile(0.90)):.4f}")

    if active_unc and active_unc in centroids.columns:
        st.markdown('<div class="section-hdr">☁️ Uncertainty distribution</div>', unsafe_allow_html=True)
        unc_vals = pd.to_numeric(centroids[active_unc], errors="coerce").dropna()
        fig_unc = px.histogram(unc_vals, nbins=40,
                               color_discrete_sequence=["#f59e0b"],
                               template="plotly_dark",
                               labels={"value": "Uncertainty"})
        fig_unc.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_unc, use_container_width=True)

    st.markdown('<div class="section-hdr">🔵 Built-up vs Vegetation scatter</div>', unsafe_allow_html=True)
    if all(c in centroids.columns for c in ["observed_2021_built_up", "observed_2021_vegetation"]):
        # sample to keep it fast
        sample = centroids.sample(min(2000, len(centroids)), random_state=42)
        color_col = "ensemble_abs_total_change" if "ensemble_abs_total_change" in sample.columns else None
        fig_scatter = px.scatter(
            sample,
            x="observed_2021_built_up",
            y="observed_2021_vegetation",
            color=color_col,
            color_continuous_scale="RdYlGn_r",
            template="plotly_dark",
            labels={
                "observed_2021_built_up":    "Built-up 2021",
                "observed_2021_vegetation":  "Vegetation 2021",
                "ensemble_abs_total_change": "Total change",
            },
            opacity=0.5,
            height=300,
        )
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)


# ---- tab 4: interpretation ----

with tab4:
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-hdr">✅ Helpful explanation</div>', unsafe_allow_html=True)
        st.success("""
A **red grid cell** in ensemble built-up change means the model expects
the built-up share of that cell to **increase by 2022** relative to 2021.
It does not mean every parcel will be built-up — it is a grid-level
proportion estimate with associated uncertainty.
""")
        st.markdown('<div class="section-hdr">⚠️ Potentially misleading</div>', unsafe_allow_html=True)
        st.warning("""
*"This cell will definitely be urbanized in 2022."*

This is misleading because the model outputs **grid-level proportions with uncertainty**,
not certain parcel-by-parcel outcomes. Small predicted changes may also fall within the
label noise between ESA WorldCover 2020 and 2021 map versions.
""")

    with right:
        st.markdown('<div class="section-hdr">📋 Product limitations</div>', unsafe_allow_html=True)
        st.markdown(f"""
- Predictions are at **{CELL_SIZE_M}m grid cell** level — not individual parcels
- Labels come from **ESA WorldCover** — any errors in source labels propagate
- The model was trained on **2020→2021** and applied to forecast **2021→2022**
- Intended for **screening and exploration only** — not for legal or enforcement use

**Intended users:** {', '.join(cfg['reporting'].get('intended_users', []))}

**Do not use for:** {', '.join(cfg['reporting'].get('not_for_decisions', []))}
""")
        st.markdown('<div class="section-hdr">🔬 Model info</div>', unsafe_allow_html=True)
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Final model | `{cfg['modeling']['app_final_model_name']}` |
| Cell size | `{CELL_SIZE_M}m` |
| Anchor year | `{cfg['modeling']['app_inference_anchor_year']}` |
| Forecast year | `{cfg['modeling']['app_forecast_year']}` |
| Change threshold | `{cfg['modeling']['change_threshold']}` |
| Spatial block | `{cfg['modeling']['spatial_block_size_m']}m` |
""")


# ---- tab 5: area search ----

with tab5:
    st.markdown('<div class="section-hdr">🔍 Search a Nuremberg neighbourhood</div>', unsafe_allow_html=True)
    st.caption("Pick a neighbourhood to highlight its grid cells and see a land cover breakdown.")

    # rough centroids for each district — good enough for a radius search
    AREAS = {
        "Altstadt (Old Town)":         (49.4537, 11.0784, 0.008),
        "Gostenhof":                   (49.4530, 11.0550, 0.007),
        "Wöhrder Wiese":               (49.4580, 11.0950, 0.007),
        "Hauptbahnhof (Main Station)": (49.4455, 11.0822, 0.006),
        "St. Lorenz":                  (49.4480, 11.0790, 0.007),
        "Maxfeld":                     (49.4650, 11.0820, 0.007),
        "Gleißhammer":                 (49.4420, 11.1100, 0.007),
        "Langwasser":                  (49.4080, 11.1050, 0.010),
        "Gibitzenhof":                 (49.4280, 11.0780, 0.007),
        "Schweinau":                   (49.4380, 11.0480, 0.007),
        "St. Johannis":                (49.4680, 11.0600, 0.008),
        "Thon":                        (49.4850, 11.0700, 0.009),
        "Zerzabelshof":                (49.4400, 11.1200, 0.007),
        "Mögeldorf":                   (49.4530, 11.1300, 0.009),
        "Lichtenhof":                  (49.4320, 11.0850, 0.007),
        "Rennweg":                     (49.4600, 11.0900, 0.007),
        "Steinbühl":                   (49.4380, 11.0680, 0.007),
        "Nordstadt":                   (49.4700, 11.0750, 0.008),
        "Gartenstadt":                 (49.4780, 11.1050, 0.009),
        "Flughafen area":              (49.4930, 11.0780, 0.010),
    }

    search_col, radius_col = st.columns([2, 1])
    with search_col:
        selected_area = st.selectbox("📍 Select neighbourhood", list(AREAS.keys()))
    with radius_col:
        radius_km = st.slider("Search radius (km)", 0.3, 2.0, 0.7, 0.1)

    area_lat, area_lon, _ = AREAS[selected_area]
    radius_deg = radius_km / 111.0

    centroids["dist_to_area"] = np.sqrt(
        (centroids["lat"] - area_lat) ** 2 + (centroids["lon"] - area_lon) ** 2
    )
    area_cells = centroids[centroids["dist_to_area"] <= radius_deg].copy()

    if len(area_cells) == 0:
        st.warning("No grid cells found in this area — try increasing the search radius.")
    else:
        st.success(f"Found **{len(area_cells)}** grid cells in **{selected_area}**")

        u20 = safe_mean(area_cells, "observed_2020_built_up")
        u21 = safe_mean(area_cells, "observed_2021_built_up")
        v20 = safe_mean(area_cells, "observed_2020_vegetation")
        v21 = safe_mean(area_cells, "observed_2021_vegetation")
        w20 = safe_mean(area_cells, "observed_2020_water")
        w21 = safe_mean(area_cells, "observed_2021_water")
        du = u21 - u20
        dv = v21 - v20
        dw = w21 - w20

        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("🏙️ Urban 2020", f"{u20*100:.1f}%")
        s2.metric("🏙️ Urban 2021", f"{u21*100:.1f}%", f"{du*100:+.2f}%")
        s3.metric("🌿 Veg 2020",   f"{v20*100:.1f}%")
        s4.metric("🌿 Veg 2021",   f"{v21*100:.1f}%", f"{dv*100:+.2f}%")
        s5.metric("💧 Water 2020", f"{w20*100:.1f}%")
        s6.metric("💧 Water 2021", f"{w21*100:.1f}%", f"{dw*100:+.2f}%")

        st.markdown("---")
        map_col, chart_col = st.columns([3, 2])

        with map_col:
            st.markdown('<div class="section-hdr">🗺️ Highlighted area</div>', unsafe_allow_html=True)

            all_poly = gdf.to_crs("EPSG:4326").copy()
            all_poly["centroid_lat"] = all_poly.geometry.centroid.y
            all_poly["centroid_lon"] = all_poly.geometry.centroid.x
            all_poly["dist"] = np.sqrt(
                (all_poly["centroid_lat"] - area_lat) ** 2
                + (all_poly["centroid_lon"] - area_lon) ** 2
            )
            all_poly["in_area"] = all_poly["dist"] <= radius_deg

            def make_area_color(row):
                if row["in_area"]:
                    u = float(row.get("observed_2021_built_up", 0) or 0)
                    v = float(row.get("observed_2021_vegetation", 0) or 0)
                    w = float(row.get("observed_2021_water", 0) or 0)
                    return [int(min(255, u*255*2.5)), int(min(255, v*255*1.2)), int(min(255, w*255*4.0)), 230]
                return [30, 30, 30, 60]

            all_poly["color"] = all_poly.apply(make_area_color, axis=1)

            highlight_layer = pdk.Layer(
                "GeoJsonLayer",
                data=all_poly.__geo_interface__,
                pickable=True, stroked=True, filled=True,
                get_fill_color="properties.color",
                get_line_color=[20, 20, 20, 60],
                line_width_min_pixels=1,
                opacity=0.9,
            )
            circle_layer = pdk.Layer(
                "ScatterplotLayer",
                data=[{"lat": area_lat, "lon": area_lon}],
                get_position="[lon, lat]",
                get_radius=radius_km * 1000,
                get_fill_color=[255, 255, 255, 8],
                get_line_color=[255, 255, 255, 180],
                stroked=True, filled=True,
                line_width_min_pixels=2,
            )
            area_tooltip = """
            <div style='background:#1a1a2e;color:#f0f0f0;padding:10px;
                        border-radius:8px;font-family:monospace;font-size:11px;
                        border:1px solid #3a3a5a'>
            <b style='color:#e8593c'>📍 {grid_id}</b><br>
            🏙️ Urban 2021: <b>{observed_2021_built_up}</b><br>
            🌿 Veg 2021: <b>{observed_2021_vegetation}</b><br>
            💧 Water 2021: <b>{observed_2021_water}</b><br>
            Δ Built-up: <b>{ensemble_delta_2022_built_up}</b>
            </div>
            """
            st.pydeck_chart(pdk.Deck(
                layers=[highlight_layer, circle_layer],
                initial_view_state=pdk.ViewState(latitude=area_lat, longitude=area_lon, zoom=13, pitch=0),
                map_style=BASEMAPS[basemap_name],
                map_provider="mapbox",
                api_keys={"mapbox": MAPBOX_TOKEN},
                tooltip={"html": area_tooltip},
            ), use_container_width=True, height=420, key=f"area_map_{basemap_name}_{selected_area}")

        with chart_col:
            st.markdown('<div class="section-hdr">📊 2021 composition</div>', unsafe_allow_html=True)
            other_21 = max(0, 1 - u21 - v21 - w21)
            donut = go.Figure(go.Pie(
                labels=["🏙️ Built-up", "🌿 Vegetation", "💧 Water", "Other"],
                values=[u21, v21, w21, other_21],
                hole=0.55,
                marker_colors=["#e8593c", "#2d9e5f", "#3b82f6", "#888888"],
                textinfo="label+percent",
                textfont_size=11,
            ))
            donut.update_layout(
                template="plotly_dark", height=240,
                margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                annotations=[dict(text="2021", x=0.5, y=0.5, font_size=16, showarrow=False, font_color="white")],
            )
            st.plotly_chart(donut, use_container_width=True)

            st.markdown('<div class="section-hdr">📉 Change 2020→2021</div>', unsafe_allow_html=True)
            bar_change = go.Figure(go.Bar(
                x=["Built-up", "Vegetation", "Water"],
                y=[du*100, dv*100, dw*100],
                marker_color=[
                    "#e8593c" if du >= 0 else "#3b82f6",
                    "#2d9e5f" if dv >= 0 else "#e8593c",
                    "#3b82f6" if dw >= 0 else "#888",
                ],
                text=[f"{du*100:+.2f}%", f"{dv*100:+.2f}%", f"{dw*100:+.2f}%"],
                textposition="outside",
            ))
            bar_change.update_layout(template="plotly_dark", height=220,
                                     margin=dict(l=0, r=0, t=10, b=0),
                                     showlegend=False, yaxis_title="Change %")
            st.plotly_chart(bar_change, use_container_width=True)

            if "forecast_2022_built_up" in area_cells.columns:
                st.markdown('<div class="section-hdr">🔮 2022 forecast</div>', unsafe_allow_html=True)
                fu = safe_mean(area_cells, "forecast_2022_built_up")
                fv = safe_mean(area_cells, "forecast_2022_vegetation")
                fw = safe_mean(area_cells, "forecast_2022_water")
                fa, fb, fc = st.columns(3)
                fa.metric("🏙️", f"{fu*100:.1f}%", f"{(fu-u21)*100:+.2f}%")
                fb.metric("🌿", f"{fv*100:.1f}%", f"{(fv-v21)*100:+.2f}%")
                fc.metric("💧", f"{fw*100:.1f}%", f"{(fw-w21)*100:+.2f}%")

        st.markdown('<div class="section-hdr">📋 Cell detail</div>', unsafe_allow_html=True)
        wanted_cols = [
            "grid_id",
            "observed_2020_built_up", "observed_2021_built_up",
            "observed_2020_vegetation", "observed_2021_vegetation",
            "observed_2020_water", "observed_2021_water",
            "ensemble_delta_2022_built_up", "ensemble_delta_2022_vegetation",
            "ensemble_abs_total_change", "ensemble_change_flag",
        ]
        wanted_cols = [c for c in wanted_cols if c in area_cells.columns]
        sort_by = "ensemble_abs_total_change" if "ensemble_abs_total_change" in area_cells.columns else wanted_cols[0]
        st.dataframe(area_cells[wanted_cols].sort_values(sort_by, ascending=False),
                     use_container_width=True, height=300)
        st.download_button(
            "⬇️ Download area CSV",
            data=area_cells[wanted_cols].to_csv(index=False).encode("utf-8"),
            file_name=f'{selected_area.replace(" ", "_")}_cells.csv',
            mime="text/csv",
        )


# ---- tab 6: model performance ----

with tab6:
    st.markdown('<div class="section-hdr">🤖 Model evaluation summary</div>', unsafe_allow_html=True)
    st.caption("Comparing Elastic Net vs Random Forest across the delta and t+1 prediction tasks.")

    if eval_df is not None:
        best = eval_df.loc[eval_df["macro_mae_delta"].idxmin()]
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("🏆 Best model",       f"{best['model']} ({best['task']})")
        w2.metric("📉 Best MAE",          f"{best['macro_mae_delta']:.4f}")
        w3.metric("🎯 Stability score",   f"{best['stability_score']:.3f}")
        w4.metric("❌ False change rate", f"{best['false_change_rate']*100:.1f}%")

        st.markdown("---")
        ec1, ec2 = st.columns(2)

        with ec1:
            st.markdown('<div class="section-hdr">📊 MAE comparison</div>', unsafe_allow_html=True)
            fig_mae = go.Figure()
            for task in eval_df["task"].unique():
                sub = eval_df[eval_df["task"] == task]
                fig_mae.add_trace(go.Bar(
                    name=task,
                    x=sub["model"],
                    y=sub["macro_mae_delta"],
                    text=[f"{v:.4f}" for v in sub["macro_mae_delta"]],
                    textposition="outside",
                    marker_color="#e8593c" if task == "delta" else "#3b82f6",
                    opacity=0.85,
                ))
            fig_mae.update_layout(barmode="group", template="plotly_dark",
                                   height=300, margin=dict(l=0, r=0, t=10, b=0),
                                   yaxis_title="Macro MAE",
                                   legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_mae, use_container_width=True)

        with ec2:
            st.markdown('<div class="section-hdr">🎯 Stability vs False change rate</div>', unsafe_allow_html=True)
            fig_ev = px.scatter(
                eval_df,
                x="false_change_rate",
                y="stability_score",
                color="model",
                symbol="task",
                text=eval_df["model"] + " / " + eval_df["task"],
                color_discrete_map={"random_forest": "#e8593c", "elastic_net": "#3b82f6"},
                template="plotly_dark",
                height=300,
                labels={"false_change_rate": "False change rate", "stability_score": "Stability score"},
            )
            fig_ev.update_traces(textposition="top center", marker_size=12)
            fig_ev.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                  legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_ev, use_container_width=True)

        st.markdown('<div class="section-hdr">📋 Full evaluation table</div>', unsafe_allow_html=True)

        # color code cells based on how good/bad the value is
        def style_mae(val):
            try:
                v = float(val)
                if v < 0.005: return "background-color:#2d9e5f33;color:#2d9e5f"
                if v < 0.008: return "background-color:#f59e0b33;color:#f59e0b"
                return "background-color:#e8593c33;color:#e8593c"
            except:
                return ""

        def style_stability(val):
            try:
                v = float(val)
                if v > 0.93: return "background-color:#2d9e5f33;color:#2d9e5f"
                if v > 0.92: return "background-color:#f59e0b33;color:#f59e0b"
                return "background-color:#e8593c33;color:#e8593c"
            except:
                return ""

        def style_fcr(val):
            try:
                v = float(val)
                if v < 0.05: return "background-color:#2d9e5f33;color:#2d9e5f"
                if v < 0.20: return "background-color:#f59e0b33;color:#f59e0b"
                return "background-color:#e8593c33;color:#e8593c"
            except:
                return ""

        styled_table = (
            eval_df.style
            .applymap(style_mae,       subset=["macro_mae_delta", "macro_mae_next_year"])
            .applymap(style_stability, subset=["stability_score"])
            .applymap(style_fcr,       subset=["false_change_rate"])
            .format({
                "macro_mae_delta":     "{:.4f}",
                "macro_mae_next_year": "{:.4f}",
                "false_change_rate":   "{:.3f}",
                "stability_score":     "{:.4f}",
            })
        )
        st.dataframe(styled_table, use_container_width=True)

        st.markdown('<div class="section-hdr">💡 What these metrics mean</div>', unsafe_allow_html=True)
        mm1, mm2, mm3 = st.columns(3)
        with mm1:
            st.info("**Macro MAE** — average absolute error across all land cover classes. Lower is better. Random Forest wins on both tasks.")
        with mm2:
            st.warning("**False change rate** — how often the model flags a cell as changed when it actually wasn't. Elastic Net is much more conservative here.")
        with mm3:
            st.success("**Stability score** — share of cells correctly identified as stable (no meaningful change). Higher is better.")

    else:
        st.error("dual_evaluation_summary.csv not found. Check the artifacts/evaluation folder.")

    # uncertainty deep dive using the separate uncertainty parquet
    if unc_df is not None:
        st.markdown("---")
        st.markdown('<div class="section-hdr">📐 Uncertainty — prediction interval widths</div>', unsafe_allow_html=True)
        st.caption(f"Based on {len(unc_df):,} test cells. Shows how wide the p10–p90 interval is per class.")

        uc1, uc2 = st.columns(2)

        with uc1:
            st.markdown('<div class="section-hdr">Built-up uncertainty</div>', unsafe_allow_html=True)
            if all(c in unc_df.columns for c in [
                "direct_uncertainty_p10_delta_built_up",
                "direct_uncertainty_p90_delta_built_up",
            ]):
                unc_df["iw_built_up"] = (
                    unc_df["direct_uncertainty_p90_delta_built_up"]
                    - unc_df["direct_uncertainty_p10_delta_built_up"]
                )
                fig_u1 = px.histogram(unc_df["iw_built_up"], nbins=40,
                                      color_discrete_sequence=["#e8593c"],
                                      template="plotly_dark",
                                      labels={"value": "p90 − p10 interval width"})
                fig_u1.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                st.plotly_chart(fig_u1, use_container_width=True)

        with uc2:
            st.markdown('<div class="section-hdr">Vegetation uncertainty</div>', unsafe_allow_html=True)
            if all(c in unc_df.columns for c in [
                "direct_uncertainty_p10_delta_vegetation",
                "direct_uncertainty_p90_delta_vegetation",
            ]):
                unc_df["iw_veg"] = (
                    unc_df["direct_uncertainty_p90_delta_vegetation"]
                    - unc_df["direct_uncertainty_p10_delta_vegetation"]
                )
                fig_u2 = px.histogram(unc_df["iw_veg"], nbins=40,
                                      color_discrete_sequence=["#2d9e5f"],
                                      template="plotly_dark",
                                      labels={"value": "p90 − p10 interval width"})
                fig_u2.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                st.plotly_chart(fig_u2, use_container_width=True)

        ua, ub, uc_col = st.columns(3)
        ua.metric("Avg built-up uncertainty",    f"{unc_df['direct_uncertainty_std_delta_built_up'].mean():.4f}")
        ub.metric("Avg vegetation uncertainty",  f"{unc_df['direct_uncertainty_std_delta_vegetation'].mean():.4f}")
        uc_col.metric("Avg water uncertainty",   f"{unc_df['direct_uncertainty_std_delta_water'].mean():.4f}")

    else:
        st.info("random_forest_uncertainty.parquet not found — skipping uncertainty section.")


st.markdown("---")
st.caption("NürnbergLens · UTN ML Final Project · ESA WorldCover + Sentinel-2 · Not for regulatory use")
