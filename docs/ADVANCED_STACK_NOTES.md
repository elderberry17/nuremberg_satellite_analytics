# Advanced stack notes

This revamp standardizes the whole pipeline around a single schema and a single config contract.

## Main improvements

- Consistent `grid_id`, `centroid_x`, `centroid_y`, `area_m2`, `change_flag`, `abs_total_change`
- Grid generation avoids clipped sliver polygons by default
- WorldCover and Sentinel stages use raster windows instead of full-raster masking
- Sentinel stage gracefully uses SWIR only when it is valid raw data
- Training artifacts store the model plus feature and target metadata
- App prediction export is aligned to the trained artifact schema
- Streamlit app auto-discovers prediction and uncertainty columns

## Expected execution order

```bash
python src/01_build_grid.py
python src/02_prepare_worldcover_labels.py
python src/03_extract_sentinel_features.py
python src/04_build_modeling_table.py
python src/05_train_models.py
python src/06_evaluate_models.py
python src/07_generate_app_predictions.py
streamlit run app/streamlit_app.py
```
