from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
PRED_PATH = BASE_DIR / 'artifacts' / 'predictions' / 'app_predictions.geojson'

st.set_page_config(page_title='Nuremberg Urban Change Intelligence', page_icon='🛰️', layout='wide')
st.title('🛰️ Nuremberg Urban Change Intelligence')
st.caption('Advanced Streamlit demo for grid-level land-cover change prediction, hotspot screening, and uncertainty-aware exploration.')


@st.cache_data
def load_geojson(path: Path):
    if not path.exists():
        return None
    return gpd.read_file(path)


def infer_metric_label(col: str) -> str:
    return col.replace('pred_', '').replace('_', ' ').title()


def find_uncertainty_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if 'uncertainty_' in c.lower() or 'std' in c.lower()]


def prepare_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.to_crs('EPSG:4326').copy()
    out['geometry'] = out.geometry.centroid
    out['lon'] = out.geometry.x
    out['lat'] = out.geometry.y
    return out


def color_signed(values: pd.Series):
    max_abs = float(np.nanmax(np.abs(values))) if len(values) else 1.0
    max_abs = max(max_abs, 1e-6)
    colors = []
    for v in values:
        if pd.isna(v):
            colors.append([180, 180, 180, 70])
            continue
        intensity = int(min(255, 45 + 210 * abs(v) / max_abs))
        colors.append([220, 70, 70, intensity] if v >= 0 else [60, 110, 230, intensity])
    return colors, max_abs


def color_magnitude(values: pd.Series):
    max_val = float(np.nanmax(values)) if len(values) else 1.0
    max_val = max(max_val, 1e-6)
    colors = []
    for v in values:
        if pd.isna(v):
            colors.append([180, 180, 180, 70])
            continue
        intensity = int(min(255, 45 + 210 * float(v) / max_val))
        colors.append([245, 140, 25, intensity])
    return colors, max_val


def hotspot_table(df: pd.DataFrame, metric: str, top_n: int, uncertainty_col: str | None, absolute_rank: bool) -> pd.DataFrame:
    out = df.copy()
    out['ranking_score'] = out[metric].abs() if absolute_rank else out[metric]
    cols = ['grid_id', metric, 'ranking_score']
    if uncertainty_col and uncertainty_col in out.columns:
        cols.append(uncertainty_col)
    for c in ['abs_total_change', 'change_flag', 'dominant_change_class']:
        if c in out.columns:
            cols.append(c)
    return out.sort_values('ranking_score', ascending=False)[cols].head(top_n).reset_index(drop=True)


gdf = load_geojson(PRED_PATH)
if gdf is None:
    st.error('Prediction file not found. Run src/07_generate_app_predictions.py first.')
    st.stop()
if gdf.crs is None:
    st.error('Prediction GeoJSON has no CRS.')
    st.stop()

prediction_cols = [c for c in gdf.columns if c.startswith('pred_')]
uncertainty_cols = find_uncertainty_columns(gdf)
if not prediction_cols:
    st.error('No prediction columns found in app_predictions.geojson.')
    st.stop()

with st.sidebar:
    st.header('Controls')
    metric = st.selectbox('Prediction layer', prediction_cols, format_func=infer_metric_label)
    map_style_name = st.selectbox('Basemap', ['Light', 'Dark', 'Satellite'], index=0)
    map_style_lookup = {'Light': 'mapbox://styles/mapbox/light-v9', 'Dark': 'mapbox://styles/mapbox/dark-v10', 'Satellite': 'mapbox://styles/mapbox/satellite-streets-v12'}
    color_mode = st.radio('Color mode', ['Signed value', 'Absolute magnitude'], index=0)
    show_unc = st.checkbox('Use uncertainty for marker size', value=bool(uncertainty_cols))
    uncertainty_col = st.selectbox('Uncertainty field', ['None'] + uncertainty_cols) if show_unc else 'None'
    top_n = st.slider('Hotspot rows', min_value=10, max_value=100, value=25, step=5)
    absolute_rank = st.checkbox('Rank by absolute magnitude', value=True)

centroids = prepare_centroids(gdf)
centroids[metric] = pd.to_numeric(centroids[metric], errors='coerce').fillna(0)
if color_mode == 'Signed value':
    centroids['color'], scale = color_signed(centroids[metric])
    legend = f'Signed color scale max |value| = {scale:.4f}'
else:
    centroids['metric_abs'] = centroids[metric].abs()
    centroids['color'], scale = color_magnitude(centroids['metric_abs'])
    legend = f'Absolute color scale max = {scale:.4f}'

centroids['radius'] = 140
if show_unc and uncertainty_col != 'None' and uncertainty_col in centroids.columns:
    unc_vals = pd.to_numeric(centroids[uncertainty_col], errors='coerce').fillna(0)
    max_unc = max(float(unc_vals.max()), 1e-6)
    centroids['radius'] = 60 + 320 * unc_vals / max_unc
else:
    uncertainty_col = None

k1, k2, k3, k4 = st.columns(4)
k1.metric('Selected layer', infer_metric_label(metric))
k2.metric('Mean', f"{float(centroids[metric].mean()):.4f}")
k3.metric('Median', f"{float(centroids[metric].median()):.4f}")
k4.metric('Range', f"{float(centroids[metric].min()):.4f} to {float(centroids[metric].max()):.4f}")
st.markdown(f'**Legend:** {legend}')

layer = pdk.Layer('ScatterplotLayer', data=centroids, get_position='[lon, lat]', get_fill_color='color', get_radius='radius', pickable=True, opacity=0.68)
view_state = pdk.ViewState(latitude=float(centroids['lat'].mean()), longitude=float(centroids['lon'].mean()), zoom=10.6, pitch=0)

tooltip_html = f"<b>Grid ID:</b> {{grid_id}}<br/><b>{metric}:</b> {{{metric}}}"
if uncertainty_col:
    tooltip_html += f"<br/><b>{uncertainty_col}:</b> {{{uncertainty_col}}}"
for c in ['abs_total_change', 'change_flag', 'dominant_change_class']:
    if c in centroids.columns:
        tooltip_html += f"<br/><b>{c}:</b> {{{c}}}"

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=map_style_lookup[map_style_name], tooltip={'html': tooltip_html, 'style': {'backgroundColor': '#1f4e79', 'color': 'white'}}), use_container_width=True)

left, right = st.columns([1.2, 1])
with left:
    st.subheader('Hotspot ranking')
    hot = hotspot_table(centroids, metric, top_n, uncertainty_col, absolute_rank)
    st.dataframe(hot, use_container_width=True)
    st.download_button('Download hotspot CSV', data=hot.to_csv(index=False).encode('utf-8'), file_name='hotspots.csv', mime='text/csv')
with right:
    st.subheader('Interpretation guide')
    mean_val = float(centroids[metric].mean())
    direction = 'positive on average' if mean_val > 0 else 'negative on average' if mean_val < 0 else 'balanced around zero'
    st.markdown(f"""
- The selected layer is **{direction}** across the city.
- Strong colors highlight the most notable predicted values.
- Larger markers indicate higher estimated uncertainty when uncertainty sizing is enabled.
- Use the hotspot table to prioritize review and discussion during the demo.
""")

st.markdown('---')
d1, d2 = st.columns(2)
with d1:
    st.subheader('Distribution diagnostics')
    hist_source = centroids[metric].dropna()
    if len(hist_source):
        counts, bins = np.histogram(hist_source, bins=30)
        hist_df = pd.DataFrame({'bin': [f'{bins[i]:.3f} to {bins[i+1]:.3f}' for i in range(len(counts))], 'count': counts}).set_index('bin')
        st.bar_chart(hist_df)
    else:
        st.info('No data available for histogram.')
with d2:
    st.subheader('Product limitations')
    st.markdown("""
- Predictions are made on **250 m grid cells**, not parcel-level land use.
- Labels come from **ESA WorldCover**, so source-label errors can propagate.
- The current production run emphasizes **RGBNIR-derived statistics and indices**; valid SWIR bands are used only when present as raw data.
- Change between **2020 and 2021** can be subtle and noisy.
- This tool is intended for **screening, exploration, and prioritization**, not legal or cadastral decisions.
""")
