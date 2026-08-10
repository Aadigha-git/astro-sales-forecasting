# Sales Forecasting MLOps using Astro

## Table of Contents
1. [Overview](#overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Technology Stack](#technology-stack)
4. [Component Details](#component-details)
5. [Data Flow](#data-flow)
6. [ML Pipeline](#ml-pipeline)
7. [Deployment Architecture](#deployment-architecture)
8. [Inference System](#inference-system)
9. [Workforce Staffing](#workforce-staffing)
10. [System Integration](#system-integration)

## Overview

This project demonstrates a production-ready MLOps pipeline for sales forecasting using Apache Airflow (via Astronomer), MLflow, MinIO, and modern machine learning techniques. Training combines tabular gradient-boosting models with classical time-series baselines, evaluates them with rolling-origin and multi-horizon protocols, and applies automated forecast quality gates before registering results. A Streamlit UI supports interactive forecasting, staffing conversion, and metric comparison.

### Key Features
- **Automated ML Pipeline**: End-to-end automation from data ingestion to model registration
- **Hybrid Modeling**: XGBoost, LightGBM, optional Prophet, plus Seasonal Naive / Holt-Winters / SARIMAX
- **Rigorous Evaluation**: Rolling-origin CV, multi-horizon (1/3/7/14 day) accuracy and bias, WAPE
- **Forecast Quality Gates**: Configurable PASS / WARNING / FAIL checks logged to MLflow
- **Experiment Tracking**: Metrics, parameters, charts, and model artifacts in MLflow + MinIO
- **Model Registry**: Versioned models with promotion workflows
- **Deployable Ensemble**: Inference-ready weighted blend of XGBoost + LightGBM
- **Interactive Inference**: Streamlit pages for forecast, staffing plan, and model comparison
- **Astronomer Integration**: Local Astro CLI development stack with Docker Compose sidecars

## Architecture Diagrams

### High-Level System Architecture
![System Architecture](system_architecture.png)

### Component Interaction Diagram
![img.png](img.png)

### Deployment Architecture
![Deployment Architecture](deployment_architecture.png)

### Data Flow Through the System
![Data Flow](data_flow_diagram.png)

### ML Pipeline Flow
![ML Pipeline](ml_pipeline_flow.png)

### Inference System Architecture
![Inference Flow](inference_flow_diagram.png)

## Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | Astronomer (Apache Airflow) | Workflow orchestration (`sales_forecast_training` DAG) |
| **ML Tracking** | MLflow 2.9+ | Experiment tracking, model registry, artifact metadata |
| **Object Storage** | MinIO | S3-compatible artifact storage |
| **Containerization** | Docker & Docker Compose | Local stack (Airflow + MLflow + MinIO + Streamlit) |
| **Tabular ML** | XGBoost, LightGBM | Feature-based gradient boosting |
| **Classical TS** | statsmodels (Holt-Winters, SARIMAX), Seasonal Naive | Baselines and CV members |
| **Optional TS** | Prophet | Configurable; disabled by default in `ml_config.yaml` |
| **Data Processing** | Pandas, NumPy | Feature pipelines and series aggregation |
| **Hyperparameter Tuning** | Optuna | Bayesian optimization for boosting models |
| **Visualization** | Matplotlib, Seaborn, Plotly | Training charts + Streamlit interactivity |
| **Inference UI** | Streamlit | Forecast, staffing, comparison |
| **Workforce** | Erlang C / occupancy (`include/workforce`) | Agents from volume + AHT |

## Component Details

### 1. Feature Engineering Pipeline
![img_2.png](img_2.png)

The feature engineering pipeline transforms raw sales data into rich features for tabular models:

**Components:**
- **Time Features**: Day of week, month, quarter, holidays, weekend flags
- **Lag Features**: Previous sales values (1, 2, 3, 7, 14, 21, 30 days)
- **Rolling Features**: Moving mean / std / min / max / median over configurable windows
- **Categorical Encoding**: Store ID and related categoricals

Classical models (Seasonal Naive, Holt-Winters, SARIMAX) operate on the target series directly rather than the engineered feature matrix.

### 2. Model Modules (`include/ml_models/`)

| Module | Role |
|--------|------|
| `train_models.py` | Orchestrates prep, training, ensemble, CV, multi-horizon, quality, visualizations |
| `classical_ts_models.py` | Sklearn-like wrappers for Seasonal Naive, Holt-Winters, SARIMAX |
| `rolling_origin_cv.py` | Expanding-window folds on train+val; fold metrics aggregation |
| `horizon_evaluation.py` | Accuracy/bias at 1/3/7/14 day horizons on daily aggregates |
| `forecast_quality.py` | Thresholded quality checks and MLflow artifacts |
| `ensemble_model.py` | Weighted `predict(X)` ensemble for deployment |
| `model_visualization.py` | Comparison charts, residuals, CV and horizon plots |
| `model_comparison.py` | Candidate vs production promotion helpers |
| `diagnostics.py` | Training diagnostics / underperformance signals |

### 3. Configuration (`include/config/ml_config.yaml`)

Key sections:
- **`models.*`**: Per-model params; classical models and Prophet use `enabled` flags
- **`training.metrics`**: rmse, mae, mape, wape, bias, r2
- **`training.rolling_origin_cv`**: n_splits, horizon, step, min_train_size
- **`training.multi_horizon_evaluation`**: horizons `[1, 3, 7, 14]`, folds, daily aggregation
- **`monitoring.forecast_quality`**: warn/fail thresholds for bias, RMSE/MAE/WAPE, negatives, timestamps
- **`workforce`**: interval length, occupancy, shrinkage, Erlang C vs occupancy method (used by staffing UI / library; not an Airflow task)

### 4. MLflow Integration
![img_3.png](img_3.png)

MLflow is the hub for experiment tracking and model management:
- **Experiment Tracking**: Metrics (including WAPE/bias and quality flattenings), parameters, tags
- **Model Registry**: Versioned model storage with stage transitions
- **Artifact Storage**: MinIO-backed plots, CV summaries, multi-horizon tables, quality reports, pickled models

## Data Flow

### Training Data Flow
1. **Raw Data Generation**: `RealisticSalesDataGenerator` (retail panel); optional `ContactCenterDemandGenerator` for hourly WFM-style volume/AHT
2. **Data Validation**: Schema and quality checks in the DAG
3. **Feature Engineering**: Time-based and aggregated features for tabular models
4. **Model Training**: Boosting (+ optional Prophet) and enabled classical baselines in parallel conceptually within `ModelTrainer.train_all_models`
5. **Evaluation**: Holdout test metrics, rolling-origin CV, multi-horizon scores, forecast quality report
6. **Artifact Storage**: Models, metrics, and visualization packages to MLflow/MinIO
7. **Registration**: Best-model signals (holdout RMSE and CV mean RMSE) feed registry/promotion tasks

## ML Pipeline

### Splits and Protocols
- **Chronological train / validation / test** split for final holdout reporting
- **Rolling-origin (expanding-window) CV** runs on **train+validation only** so the holdout test set stays untouched
- **Multi-horizon evaluation** aggregates to a daily series, then scores cumulative forecast windows at each horizon

### Models
- **XGBoost / LightGBM**: Primary deployable predictors; Optuna tuning available
- **Prophet**: Optional calendar/seasonality model when enabled
- **Seasonal Naive / Holt-Winters / SARIMAX**: Classical baselines for comparison and evaluation ensemble weighting
- **Ensemble**:
  - *Evaluation blend*: Can weight all successfully scored members (including classical) for holdout comparison metrics
  - *Deployable artifact*: `EnsembleModel` of XGBoost + LightGBM only, matching the Streamlit `predict(X)` path

### Metrics
- **Error**: RMSE, MAE, MAPE, WAPE
- **Bias**: mean(prediction − actual); positive means over-forecast
- **Fit**: R² (where defined)
- Per-model results may include `metrics` (holdout), `cv_metrics` (rolling-origin aggregates), and `horizon_metrics`

### Quality Gates
`ForecastQualityChecker` applies configurable thresholds for:
- Absolute / relative bias
- RMSE, MAE, WAPE
- Negative actual volumes or predictions
- Missing predictions / non-finite values
- Missing or duplicate timestamps vs expected frequency

Overall status is the worst of PASS / WARNING / FAIL across checks and is summarized in the DAG report and Streamlit Model Comparison page.

### Visualization Outputs
Logged training visuals typically include:
- Metrics comparison bars and comparison tables (MAE / RMSE / WAPE / Bias)
- Predictions vs actuals, residuals, error distributions
- Feature importance (tabular models)
- Rolling-origin CV summaries
- Forecast accuracy by horizon

## Deployment Architecture

### Docker-Based Deployment
The system uses Astronomer CLI + Docker Compose override for containerized local deployment:

1. **Astronomer Services**
   - Airflow webserver / API, scheduler, workers (as provided by Astro)
   - Metadata DB and supporting services from the Astro project image

2. **ML Services** (via `docker-compose.override.yml`)
   - MLflow Server (port 5001)
   - MinIO Object Storage (ports 9000 / 9001)

3. **Inference Services**
   - Streamlit UI (port 8501), volume-mounted `ui/` + `include/`

4. **Persistent Volumes**
   - MLflow / MinIO data volumes
   - Astro-managed Airflow volumes for logs and state

### Network Architecture
Services communicate on Docker networks managed by Astro and the compose override. The override may declare the Airflow network as **external** with a project-specific name; that network must exist (created by a prior Astro start) or `astro dev start` will fail until it is created or the stack is reset.

## Inference System

### Streamlit-Based Interfaces

![img_6.png](img_6.png)

Pages (Streamlit multipage under `ui/`):

1. **Forecast (`inference_app.py`)**
   - Inputs: CSV upload, 7-day manual entry, or sample data
   - Models loaded from latest MLflow run: **ensemble**, **xgboost**, **lightgbm**
   - Plotly forecast charts with confidence-style bands and CSV export

2. **Staffing Plan (`pages/1_Staffing_Plan.py`)**
   - Converts forecasted volume + AHT into offered workload, required productive/paid agents, and staffing gap
   - Assumptions loaded from `workforce` config (occupancy, shrinkage, Erlang C vs occupancy)

3. **Model Comparison (`pages/2_Model_Comparison.py`)**
   - Reads MLflow run metrics for holdout MAE/RMSE/WAPE/bias
   - Surfaces multi-horizon and forecast quality summaries when present

### Inference Boundaries
- Classical TS pickles are saved under training artifacts for analysis, but the current Streamlit loader does **not** expose Seasonal Naive / Holt-Winters / SARIMAX as selectable forecast models
- Deployable ensemble weights are renormalized over tabular members only

### Prediction Flow
```text
Historical Input → Feature Engineering / Scaling → Tabular or Ensemble Predict → Charts / CSV
```

## Workforce Staffing

`include/workforce/staffing.py` is a library (not an Airflow task) that:
- Accepts interval volume and AHT
- Computes Erlang C or occupancy-based required agents
- Applies shrinkage to estimate rostered / paid headcount
- Powers the Streamlit Staffing Plan page and can be reused from notebooks/scripts

Synthetic contact-center style series can be generated via `ContactCenterDemandGenerator` in `include/utils/data_generator.py` for staffing demos independent of the retail sales DAG path.

## System Integration

### Astronomer Platform Integration
- **Local Astro CLI**: `astro dev start` brings up Airflow and mounts project DAGs / `include/`
- **DAG**: `sales_forecast_training` — extract → validate → feature/train → evaluate → register → report
- **Dependencies**: Root `requirements.txt` / `packages.txt` bake into the Astro runtime image; Streamlit and MLflow have separate Dockerfiles

### Key Integration Points
1. **DAG Deployment**: Project sync via Astronomer CLI
2. **Shared Code**: `include/` used by Airflow tasks and (via path setup) Streamlit
3. **Config**: `ml_config.yaml` (in-container) and `ml_config_local.yaml` for local overrides
4. **Artifact Path**: Training writes under `/tmp/` then logs artifacts to MLflow → MinIO
5. **UI ↔ Tracking**: Streamlit talks to MLflow tracking URI and MinIO credentials from the compose environment
