"""Shared Streamlit session helpers (model loader + MLflow)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
import streamlit as st

from utils.path_setup import ensure_include_on_path, ml_config_path
from utils.simple_model_loader import SimpleModelLoader
from utils.simple_predictor import SimplePredictor

ensure_include_on_path()


def init_session() -> None:
    if "model_loader" not in st.session_state:
        st.session_state.model_loader = SimpleModelLoader()
        st.session_state.predictor = SimplePredictor(st.session_state.model_loader)
        st.session_state.models_loaded = False
        st.session_state.run_id = None


def render_model_loader_sidebar() -> None:
    """Common sidebar controls for loading models from MLflow."""
    init_session()
    st.sidebar.header("📦 Model Configuration")
    if not st.session_state.models_loaded:
        st.sidebar.warning("⚠️ No models loaded")
    else:
        st.sidebar.success("✅ Models loaded")
        st.sidebar.info(
            f"Models: {', '.join(st.session_state.model_loader.models.keys())}"
        )
        if st.session_state.run_id:
            st.sidebar.caption(f"Run ID: {st.session_state.run_id[:8]}...")

    if st.sidebar.button("🔄 Load/Reload Models", type="primary", use_container_width=True):
        with st.spinner("Loading models..."):
            run_id = st.session_state.model_loader.get_latest_run()
            if run_id and st.session_state.model_loader.load_models_from_run(run_id):
                st.session_state.models_loaded = True
                st.session_state.run_id = run_id
                st.sidebar.success("✅ Models loaded!")
                st.rerun()
            else:
                st.sidebar.error("❌ Failed to load models — run the training DAG first")


def fetch_run_metrics(run_id: Optional[str] = None) -> tuple:
    """Pull flat metrics dict for the latest (or given) MLflow run.

    Returns (metrics_dict, run_id).
    """
    tracking = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking)
    client = mlflow.tracking.MlflowClient()
    if not run_id:
        exp = client.get_experiment_by_name("sales_forecasting")
        if exp is None:
            return {}, None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return {}, None
        run_id = runs[0].info.run_id
    run = client.get_run(run_id)
    return dict(run.data.metrics), run_id


def metrics_to_model_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Convert flat MLflow metrics like ``xgboost_rmse`` into a comparison table
    with columns model, mae, rmse, wape, bias, mape, r2.
    """
    models: Dict[str, Dict[str, float]] = {}
    suffixes = ("rmse", "mae", "mape", "wape", "bias", "r2")
    for key, value in metrics.items():
        if key.startswith("fq_"):
            continue
        if "_cv_" in key:
            continue
        parts = key.split("_")
        if len(parts) >= 3 and parts[-2].startswith("h") and parts[-2][1:].isdigit():
            continue
        for suffix in suffixes:
            token = f"_{suffix}"
            if key.endswith(token):
                model = key[: -len(token)]
                if model.endswith("_mean") or model.endswith("_std"):
                    continue
                models.setdefault(model, {})[suffix] = float(value)
                break
    if not models:
        return pd.DataFrame()
    rows = [{"model": m, **vals} for m, vals in models.items()]
    df = pd.DataFrame(rows)
    if "rmse" in df.columns:
        df = df.sort_values("rmse", na_position="last")
    return df.reset_index(drop=True)


def config_path() -> str:
    return ml_config_path()
