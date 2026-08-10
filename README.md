# Astro Sales Forecasting MLOps Platform

## Overview

A production-ready MLOps platform for sales forecasting that demonstrates modern machine learning engineering practices. Built on Astronomer (Apache Airflow), this project implements an end-to-end ML pipeline with tabular + classical time-series models, rigorous cross-validation, forecast quality gates, and real-time inference via Streamlit (including staffing and model comparison views).

### Key Features

- **Automated ML Pipeline**: End-to-end orchestration with Astronomer/Airflow
- **Hybrid Model Suite**: XGBoost, LightGBM, optional Prophet, plus classical baselines (Seasonal Naive, Holt-Winters, SARIMAX)
- **Time-Series Evaluation**: Rolling-origin CV, multi-horizon accuracy (1/3/7/14 day), WAPE and bias metrics
- **Forecast Quality Gates**: Automated PASS / WARNING / FAIL checks on bias, error, negatives, and timestamp integrity
- **Deployable Ensemble**: Weighted XGBoost + LightGBM for inference (classical models participate in evaluation / weighting comparisons)
- **Streamlit Apps**: Forecast inference, staffing plan (Erlang/occupancy), and model comparison from MLflow metrics
- **Experiment Tracking**: MLflow for model versioning, metrics, and visualization artifacts
- **Distributed Storage**: MinIO S3-compatible object storage for artifacts
- **Containerized Deployment**: Docker-based architecture for consistency

## Architecture

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | Astronomer (Airflow) | Workflow automation and scheduling |
| **ML Tracking** | MLflow 2.9+ | Experiment tracking and model registry |
| **Storage** | MinIO | S3-compatible artifact storage |
| **Tabular Models** | XGBoost, LightGBM | Gradient boosting on engineered features |
| **Classical TS** | Seasonal Naive, Holt-Winters, SARIMAX | Baselines and evaluation members (`statsmodels`) |
| **Visualization** | Matplotlib, Seaborn, Plotly | Model analysis and Streamlit charts |
| **Inference UI** | Streamlit | Forecast, staffing, and comparison interfaces |
| **Workforce** | Erlang C / occupancy | Staffing from volume + AHT forecasts |
| **Containerization** | Docker & Docker Compose | Environment consistency |

## Quick Start

### Prerequisites

- Docker Desktop installed and running
- Astronomer CLI (`brew install astro` on macOS; other OS: [install guide](https://www.astronomer.io/docs/astro/cli/install-cli/))
- 8GB+ RAM available for Docker
- Ports 8080, 8501, 5001, 9000, 9001 available

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/airscholar/astro-salesforecast.git
cd Astro-SalesForecast
```

### 2. Start All Services

```bash
# Start Astronomer Airflow + sidecars (MLflow, MinIO, Streamlit)
astro dev start
```

**Image caching:** Streamlit and MLflow use baked Docker images (`astro-salesforecast-streamlit`, `astro-salesforecast-mlflow`). Dependencies install **once at image build**. App code under `ui/` and `include/` is volume-mounted, so normal Python edits do **not** rebuild images. Rebuild only when `ui/requirements.txt`, `ui/Dockerfile.streamlit`, `docker/mlflow/Dockerfile`, root `Dockerfile`, `requirements.txt`, or `packages.txt` change:

```bash
astro dev stop
docker compose -f docker-compose.override.yml build streamlit-ui mlflow
astro dev start
```

This will start:
- **Airflow UI**: printed by the CLI (e.g. `http://astro-salesforecast.localhost:6563`) — admin/admin
- **Streamlit UI**: http://localhost:8501 (Forecast, Staffing Plan, Model Comparison)
- **MLflow UI**: http://localhost:5001
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)

### 3. Run the ML Pipeline

1. Open the Airflow UI (URL from `astro dev start`)
2. Enable the `sales_forecast_training` DAG
3. Trigger the DAG manually or wait for the scheduled weekly run
4. Monitor progress in the Airflow UI; charts and quality reports land in MLflow/MinIO

### 4. Use the Inference UI

1. Open Streamlit at http://localhost:8501
2. Click **Load/Reload Models** in the sidebar
3. Use pages:
   - **Forecast**: CSV / manual / sample input → XGBoost, LightGBM, or ensemble predictions
   - **Staffing Plan**: volume + AHT → workload, required agents, staffing gap
   - **Model Comparison**: holdout MAE/RMSE/WAPE/bias, multi-horizon curves, quality gate status from MLflow

## ML Pipeline Features

### Data Processing
- Synthetic retail sales data with realistic seasonality and promotions
- Optional contact-center demand generator (hourly volume / AHT by channel)
- Time-based train/validation/test splitting
- Data validation and quality checks
- Feature engineering (lags, rolling stats, calendar / holiday features)

### Model Training
- **XGBoost / LightGBM**: Gradient boosting on tabular features (Optuna tuning supported)
- **Prophet**: Optional (`models.prophet.enabled` in config; off by default)
- **Classical baselines**: Seasonal Naive, Holt-Winters, SARIMAX (enabled in `ml_config.yaml`)
- **Ensemble**: Validation-weighted blend for evaluation; deployable artifact is XGBoost + LightGBM only (compatible with Streamlit `predict(X)` API)

### Evaluation & Quality
- Holdout metrics: RMSE, MAE, MAPE, **WAPE**, **bias**, R²
- **Rolling-origin CV** on train+validation (holdout test untouched)
- **Multi-horizon evaluation** at 1 / 3 / 7 / 14 day horizons
- **Forecast quality gates** (bias, error thresholds, negative volumes/preds, missing/duplicate timestamps)
- Best-model selection reports both holdout RMSE and rolling-origin CV mean RMSE

### Visualization Suite
- Model metrics comparison (MAE / RMSE / WAPE / Bias) and comparison tables
- Predictions vs actuals, residual and error-distribution plots
- Feature importance rankings
- Rolling-origin CV fold summaries
- Forecast accuracy by horizon
- Artifacts logged to MLflow / MinIO

### Model Management
- Automated experiment tracking with MLflow
- Model versioning and registry
- Artifact storage in MinIO
- Production model promotion workflow

## Inference System

### Streamlit Features
- **Forecast page**: CSV upload, manual entry, or sample data; ensemble / XGBoost / LightGBM
- **Staffing Plan**: Erlang C or occupancy-based required agents from forecasted volume + AHT
- **Model Comparison**: Reads latest MLflow run metrics (holdout, multi-horizon, quality summary)
- Interactive Plotly charts and CSV export

> **Note:** Classical TS models are trained, compared, and logged during the DAG, but the Streamlit forecast loader currently serves the deployable tabular models (ensemble / XGBoost / LightGBM). Use **Model Comparison** (or MLflow) to inspect classical baseline metrics.

### Prediction Flow
```text
Input Data → Feature Engineering → Model Prediction → Visualization → Export
```

## Performance & Metrics

- **Training Time**: Depends on enabled models and CV (classical + multi-horizon add runtime)
- **Prediction Latency**: Sub-second for tabular / ensemble inference in Streamlit
- **Primary accuracy metrics**: RMSE, MAE, WAPE, bias (plus MAPE / R²)
- **Selection signals**: Holdout RMSE and rolling-origin CV mean RMSE

## Troubleshooting

### Common Issues

1. **Services not starting**: Check Docker memory allocation (8GB minimum)
2. **External network missing**: If you see `network ..._airflow declared as external, but could not be found`, create it (`docker network create <name>`) or re-run a clean `astro dev start` after `astro dev stop`
3. **Models not loading**: Ensure `sales_forecast_training` completed successfully
4. **Port conflicts**: Stop conflicting services or modify ports in docker-compose
5. **MLflow connection**: Verify MLflow is running and reachable from the UI container

### Logs and Debugging

```bash
# Check Airflow logs
astro dev logs

# Check specific service logs
docker-compose -f docker-compose.override.yml logs mlflow
docker-compose -f docker-compose.override.yml logs streamlit-ui
```

## Documentation

- [Detailed Architecture](docs/ARCHITECTURE.md)
- [UI README](ui/README.md)
- [Astronomer Docs](https://www.astronomer.io/docs/)

## Inspired by
https://www.youtube.com/watch?v=q74qym22vqA
