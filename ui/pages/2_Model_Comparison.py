"""
Model Comparison — holdout metrics, WAPE/bias, multi-horizon, quality gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.path_setup import ensure_include_on_path
from utils.ui_common import (
    fetch_run_metrics,
    init_session,
    metrics_to_model_table,
    render_model_loader_sidebar,
)

ensure_include_on_path()

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
init_session()
render_model_loader_sidebar()

st.title("📊 Model Comparison")
st.markdown(
    "Compare models on holdout accuracy (MAE, RMSE, WAPE, bias), "
    "multi-horizon performance, and forecast quality status from the latest MLflow run."
)

run_id = st.session_state.get("run_id")
col_a, col_b = st.columns([2, 1])
with col_a:
    st.caption("Metrics are read from MLflow experiment `sales_forecasting`.")
with col_b:
    refresh = st.button("🔄 Refresh metrics", use_container_width=True)

if refresh or "comparison_metrics" not in st.session_state:
    try:
        metrics, resolved_run = fetch_run_metrics(run_id)
        st.session_state["comparison_metrics"] = metrics
        st.session_state["comparison_run_id"] = resolved_run
    except Exception as e:
        st.error(f"Could not load MLflow metrics: {e}")
        st.info("Start the stack (`astro dev start`), run `sales_forecast_training`, then refresh.")
        st.stop()

metrics = st.session_state.get("comparison_metrics") or {}
resolved_run = st.session_state.get("comparison_run_id")
if not metrics:
    st.warning("No metrics found. Trigger the training DAG, then click Refresh.")
    st.stop()

st.success(f"Loaded metrics from run `{resolved_run}`")

# --- Holdout comparison table ---
table = metrics_to_model_table(metrics)
st.subheader("Holdout metrics (MAE / RMSE / WAPE / Bias)")
if table.empty:
    st.info("No holdout model metrics with expected naming (`{model}_rmse`, …).")
else:
    display = table.copy()
    for col in ["mae", "rmse", "wape", "bias", "mape", "r2"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    st.dataframe(display, use_container_width=True)

    # Bar chart for core metrics
    plot_df = table.melt(
        id_vars=["model"],
        value_vars=[c for c in ["mae", "rmse", "wape", "bias"] if c in table.columns],
        var_name="metric",
        value_name="value",
    )
    fig = px.bar(
        plot_df,
        x="model",
        y="value",
        color="metric",
        barmode="group",
        title="Model comparison — MAE / RMSE / WAPE / Bias",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

# --- Multi-horizon ---
st.subheader("Accuracy & bias by forecast horizon")
horizon_rows = []
for key, value in metrics.items():
    # pattern: model_h7_rmse
    parts = key.split("_")
    if len(parts) >= 3 and parts[-2].startswith("h") and parts[-2][1:].isdigit():
        horizon = int(parts[-2][1:])
        metric = parts[-1]
        model = "_".join(parts[:-2])
        if metric in {"rmse", "mae", "wape", "bias", "mape"}:
            horizon_rows.append(
                {"model": model, "horizon": horizon, "metric": metric, "value": float(value)}
            )

if not horizon_rows:
    st.info("No multi-horizon metrics yet (`{model}_h{N}_rmse`). Re-run training with multi-horizon enabled.")
else:
    hdf = pd.DataFrame(horizon_rows)
    metric_choice = st.selectbox(
        "Metric",
        [m for m in ["rmse", "mae", "wape", "bias"] if m in set(hdf["metric"])],
        key="h_metric",
    )
    sub = hdf[hdf["metric"] == metric_choice].sort_values(["model", "horizon"])
    fig_h = px.line(
        sub,
        x="horizon",
        y="value",
        color="model",
        markers=True,
        title=f"{metric_choice.upper()} vs forecast horizon (days)",
    )
    fig_h.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    fig_h.update_xaxes(dtick=1)
    if metric_choice == "bias":
        fig_h.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig_h, use_container_width=True)
    st.dataframe(
        sub.pivot(index="model", columns="horizon", values="value").round(4),
        use_container_width=True,
    )

# --- Forecast quality gate ---
st.subheader("Forecast quality gate")
fq_overall = metrics.get("fq_overall_status")
status_map = {0.0: ("PASS", "green"), 1.0: ("WARNING", "orange"), 2.0: ("FAIL", "red")}
if fq_overall is None:
    st.info("No `fq_overall_status` metric — quality checks may be disabled or not yet run.")
else:
    label, color = status_map.get(float(fq_overall), ("UNKNOWN", "gray"))
    st.markdown(f"**Overall:** :{color}[{label}]")

    fq_rows = []
    checks = [
        "missing_timestamps",
        "duplicate_timestamps",
        "negative_volumes",
        "negative_predictions",
        "missing_predictions",
        "excessive_bias",
        "abnormal_errors",
    ]
    for key, value in metrics.items():
        if not key.startswith("fq_") or key == "fq_overall_status":
            continue
        rest = key[len("fq_") :]
        if not rest.endswith("_status"):
            continue
        body = rest[: -len("_status")]
        model, check = body, "model_summary"
        for cname in checks:
            token = f"_{cname}"
            if body.endswith(token):
                model = body[: -len(token)]
                check = cname
                break
        fq_rows.append(
            {
                "model": model,
                "check": check,
                "status": status_map.get(float(value), ("?", "gray"))[0],
                "code": float(value),
            }
        )
    if fq_rows:
        st.dataframe(pd.DataFrame(fq_rows), use_container_width=True, height=280)

# --- Live side-by-side predictions (optional) ---
st.subheader("Live multi-model forecast (optional)")
if not st.session_state.models_loaded:
    st.info("Load models in the sidebar to overlay ensemble / xgboost / lightgbm forecasts.")
else:
    if st.button("Generate comparison forecast on sample history"):
        import numpy as np
        from datetime import datetime, timedelta

        hist_days = 60
        dates = pd.date_range(end=datetime.now(), periods=hist_days, freq="D")
        sales = 5000 + 300 * np.sin(2 * np.pi * np.arange(hist_days) / 7)
        hist = pd.DataFrame({"date": dates, "store_id": "store_001", "sales": sales})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["sales"], name="History", line=dict(color="#333")))
        for model_type in ["ensemble", "xgboost", "lightgbm"]:
            if model_type not in st.session_state.model_loader.models and model_type != "ensemble":
                continue
            res = st.session_state.predictor.predict(hist, model_type=model_type, forecast_days=14)
            if res.get("success"):
                pred = res["predictions"]
                fig.add_trace(
                    go.Scatter(
                        x=pred["date"],
                        y=pred["predicted_sales"],
                        name=model_type,
                        mode="lines",
                    )
                )
                # intervals for primary model only clutter — show ensemble band
                if model_type == "ensemble":
                    fig.add_trace(
                        go.Scatter(
                            x=pred["date"],
                            y=pred["upper_bound"],
                            line=dict(width=0),
                            showlegend=False,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=pred["date"],
                            y=pred["lower_bound"],
                            fill="tonexty",
                            name="ensemble interval",
                            line=dict(width=0),
                            fillcolor="rgba(46,134,171,0.2)",
                        )
                    )
        fig.update_layout(height=420, hovermode="x unified", title="14-day forecast comparison")
        st.plotly_chart(fig, use_container_width=True)
