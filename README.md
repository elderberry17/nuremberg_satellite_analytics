# UTN Machine Learning Final Project  
## Mapping Urban Change in Nuremberg with Sentinel-2, ESA WorldCover, and Explainable Tabular Machine Learning

This project is an end-to-end geospatial machine learning pipeline for analyzing and forecasting urban land-cover change in Nuremberg, Germany. The system uses ESA WorldCover as the label source and Sentinel-2 imagery as the predictor source, converts both into aligned grid-based tabular data, trains multiple machine learning models for both land-cover change and next-year composition forecasting, evaluates them under a spatial hold-out design, and serves the results through an interactive Streamlit dashboard.

The final product is designed as an MSc-level machine learning and remote sensing project with an emphasis on:
- interpretable and defensible modeling choices,
- spatially aware evaluation,
- explainability for non-expert users,
- uncertainty communication,
- and a polished interactive front end.

## Project Team

This project was developed by:

- Alexei Bezgin  
- Aryan Shams Ansari  
- Azariah Asafo Agyei  
- Emir Balci  
- Soban Mohammed Khalid  

using Python version 3.11.9 and the libraries listed in `requirements.txt`. 
The project is structured to be fully reproducible, with all data processing, modeling, evaluation, and visualization steps included in the repository.

All five members contributed equally to the project.

---

## Project Overview

The workflow represents Nuremberg as a regular 250 m grid. For each grid cell, the project:
1. computes land-cover composition labels from ESA WorldCover,
2. extracts spectral summary features and indices from Sentinel-2,
3. builds a modeling table for supervised learning,
4. trains multiple models for two forecasting formulations,
5. evaluates them with both standard and change-aware metrics,
6. generates full-coverage app predictions,
7. and visualizes the outputs in a Streamlit dashboard.

The four target land-cover classes are:
- built_up
- vegetation
- water
- other

Two prediction tasks are supported:
- **Delta prediction**: predict class-wise change between the anchor year and the target year
- **T+1 prediction**: predict next-year class composition directly

The current training pipeline supports:
- Elastic Net (Linear Regression with L1 and L2 regularization)
- Random Forest (Ensemble of Decision Trees with bootstrap aggregation)
- Gradient Boosting (Ensemble of Decision Trees with sequential boosting)

Optional hyper-parameter optimization is also supported through Optuna.

---

## Repository Structure

```text
project/
├── config/
│   └── project_config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   ├── evaluation/
│   └── paper_figures/
├── src/
│   ├── 01_build_grid.py
│   ├── 02_prepare_worldcover_labels.py
│   ├── 03_extract_sentinel_features.py
│   ├── 04_build_modeling_table.py
│   ├── 05_train_models_multi_hpo.py
│   ├── 06_evaluate_models_multi.py
│   ├── 07_generate_app_predictions_multi.py
│   ├── 08_make_paper_figures.py
│   └── common.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## Final Pipeline Stages

The workflow represents Nuremberg as a regular 250 m grid. For each grid cell, the project:
### 1. Build the spatial analysis grid

Creates the regular Nuremberg grid used as the core spatial unit of analysis.
```bash
python src/01_build_grid.py
```

### 2. Prepare ESA WorldCover labels
Processes ESA WorldCover data to compute class-wise composition labels for each grid cell and year. Extracts yearly class proportions for each grid cell and computes change labels.
```bash
python src/02_prepare_worldcover_labels.py
```

### 3. Extract Sentinel-2 features
Extracts spectral summary features and indices from Sentinel-2 imagery for each grid cell and year. Reads Sentinel SAFE scenes, extracts band summaries and spectral indices per grid cell, and saves the predictor table.
```bash
python src/03_extract_sentinel_features.py
```
### 4. Build the modeling table
Combines the features and labels into a single modeling table for supervised learning. Merges spatial features, labels, and Sentinel summaries into the final supervised learning table.
```bash
python src/04_build_modeling_table.py
```
### 5. Train machine learning models
Trains multiple models for both delta and T+1 prediction formulations, with optional hyper-parameter optimization. Trains the multi-output models for both delta and T+1 tasks using a spatial block hold-out. Optional HPO can be enabled in the config.
```bash
python src/05_train_models_multi_hpo.py
```
### 6. Evaluate the trained models
Evaluates the trained models using both standard regression metrics and change-aware metrics under a spatial hold-out design. Computes standard regression metrics, change-aware metrics, missingness stress tests, and random-forest uncertainty outputs.
```bash
python src/06_evaluate_models_multi.py
```
### 7. Generate app-ready predictions
Runs the selected trained models over all grid cells and exports the final GeoJSON for the front end.
```bash
python src/07_generate_app_predictions_multi.py
```
### 8. Generate paper figures
Creates visualizations for the final report and presentation. Creates evaluation and diagnostics figures for the technical report or IEEE paper.
```bash
python src/08_make_paper_figures.py
```
### 9. Launch the Streamlit dashboard
Serves the predictions and evaluation results through an interactive Streamlit dashboard.
```bash
streamlit run app/streamlit_app.py
```
---
## How to Run the Project

### Step 1: Create and activate a virtual environment
#### On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
#### On Linux or MacOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```
### Step 3: Verify the config paths
Open:
`config/project_config.yaml` 

and make sure the paths for:

- boundary data
- WorldCover rasters
- Sentinel SAFE folders
- processed outputs
- artifact folders

all match your local machine.

### Step 4: Run the pipeline stages in order
```bash
cd src
python 01_build_grid.py
python 02_prepare_worldcover_labels.py
python 03_extract_sentinel_features.py
python 04_build_modeling_table.py
python 05_train_models_multi_hpo.py
python 06_evaluate_models_multi.py
python 07_generate_app_predictions_multi.py
python 08_make_paper_figures.py

cd ../app
streamlit run streamlit_app.py
```

---

## Key Methodological Choices
### Grid-based analysis

The city is represented as regular spatial cells rather than parcels or raw pixels. 
This makes the problem well-structured for tabular machine learning and ensures that features, labels, predictions, and app outputs are all aligned to the same spatial unit.

### Interpretable tabular features

The predictor set consists of spectral summary statistics and indices rather than raw pixel values or complex deep features.
This allows for more interpretable models and easier debugging, while still capturing the relevant spectral information for land-cover classification.
Instead of training end-to-end image models, the project summarizes Sentinel-2 imagery into physically meaningful statistics:

- band medians,
- quartiles,
- standard deviations,
- NDVI,
- NDWI,
- NDBI.

This supports interpretability and aligns with the project’s requirement for explainable modeling.

### Dual prediction formulation

The project supports two prediction formulations:

- direct change forecasting through delta targets,
- direct next-year composition forecasting through T+1 targets.

This makes it possible to compare whether change is easier to model directly or indirectly through next-year composition.

### Spatially aware evaluation

The data are not split randomly. Instead, training and test sets are created using spatial blocks so that nearby grid cells do not leak information across the split. This produces a more realistic estimate of generalization.

### Optional hyper-parameter optimization

The training script supports optional HPO through Optuna. HPO is performed only within the outer training set by using an inner spatial validation split. The held-out outer test set remains untouched until final evaluation.

### Responsible deployment

The Streamlit app includes:

- observed and predicted layers,
- hotspot ranking,
- multiple basemaps,
- uncertainty overlays,
- interpretation guidance,
- and explicit warnings about limitations and non-decision use.

---

## Main Output Artifacts

After a full run, the project produces the following important outputs:

### Processed data
- `grid.geojson`
- `labels_table.parquet`
- `sentinel_features.parquet`
- `modeling_table.parquet`

### Training artifacts
- trained model bundles in `artifacts/models/`
- per-model metric CSVs in `artifacts/metrics/`
- prediction Parquets in `artifacts/predictions/`
- split manifest for reproducibility

### Evaluation artifacts
- `dual_evaluation_summary.csv`
- stress-test CSVs
- random-forest uncertainty table

### App artifacts
- final app prediction GeoJSON for Streamlit

### Paper/report artifacts
- evaluation plots and paper-ready figures in `artifacts/paper_figures/`

---

## Requirements

Install all required packages with:

```bash
pip install -r requirements.txt
```
The project depends on:

- python == 3.11.9
- geopandas == 1.1.3
- rasterio == 1.4.4
- numpy == 2.4.3
- pandas == 3.0.1
- scikit-learn == 1.8.0
- optuna == 4.7.0
- matplotlib == 3.10.8
- plotly == 6.6.0
- pydeck == 0.9.1
- streamlit == 1.55.0
- shapely == 2.1.2
- pyyaml == 6.0.3
- joblib == 1.5.2

If you encounter installation issues on Windows, install geospatial packages carefully and ensure GDAL-compatible wheels are available in your environment.

---

### Intended Use

This system is intended for:

- exploratory urban analysis,
- land-cover change screening,
- academic demonstration,
- model comparison and explainability,
- educational and planning-oriented visualization.

It is not intended for:

- legal boundary interpretation,
- parcel-level enforcement,
- cadastral decisions,
- or high-stakes policy decisions without further validation.

---

## Acknowledgment

This project was developed as part of the Machine Learning Course at the University of Technology Nuremberg(UTN), Germany as a Final Project and integrates remote sensing, tabular machine learning, spatial evaluation, uncertainty analysis, and interactive visualization into one reproducible end-to-end workflow.

Special thanks to the course instructors Prof. Josif Grabocka and Prof. Yuki Asano for their guidance and feedback throughout the project development process.
