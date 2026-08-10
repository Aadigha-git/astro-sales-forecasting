"""
Staffing Plan — forecasted demand → workload → required staffing → gap.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Bootstrap before other local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.path_setup import ensure_include_on_path, ml_config_path
from utils.ui_common import init_session, render_model_loader_sidebar

ensure_include_on_path()

from workforce.staffing import StaffingAssumptions, StaffingCalculator  # noqa: E402

st.set_page_config(page_title="Staffing Plan", page_icon="👥", layout="wide")
init_session()
render_model_loader_sidebar()

st.title("👥 Staffing Plan")
st.markdown(
    "Convert forecasted operational volume and average handle time into "
    "workload, required staffing, prediction intervals, and staffing gap."
)

# --- Assumptions ---
cfg_path = ml_config_path()
try:
    assumptions = StaffingAssumptions.from_yaml(cfg_path)
except Exception:
    assumptions = StaffingAssumptions()

with st.sidebar:
    st.markdown("---")
    st.header("⚙️ Staffing Assumptions")
    occupancy = st.slider("Occupancy target", 0.5, 1.0, float(assumptions.occupancy_target), 0.01)
    shrinkage = st.slider("Shrinkage", 0.0, 0.6, float(assumptions.shrinkage), 0.01)
    sl_target = st.slider("Service level target", 0.5, 1.0, float(assumptions.service_level_target), 0.01)
    answer_time = st.number_input(
        "Answer time (sec)", min_value=0, value=int(assumptions.answer_time_seconds)
    )
    method = st.selectbox(
        "Method",
        ["erlang_c", "occupancy"],
        index=0 if assumptions.method == "erlang_c" else 1,
    )
    scheduled_override = st.number_input(
        "Scheduled agents (optional gap calc)",
        min_value=0,
        value=0,
        help="If > 0, staffing gap = required − scheduled",
    )

assumptions = StaffingAssumptions(
    interval_seconds=assumptions.interval_seconds,
    occupancy_target=occupancy,
    shrinkage=shrinkage,
    service_level_target=sl_target,
    answer_time_seconds=float(answer_time),
    method=method,
    max_agents=assumptions.max_agents,
)
calc = StaffingCalculator(assumptions)

# --- Demand input ---
tab_cc, tab_sales, tab_upload = st.tabs(
    ["📞 Contact-center sample", "🛒 From sales forecast", "📤 Upload CSV"]
)

demand_df = None

with tab_cc:
    st.markdown("Generate synthetic hourly contact-center demand (volume + AHT).")
    cols = st.columns(4)
    with cols[0]:
        days = st.number_input("Days", 1, 30, 7, key="cc_days")
    with cols[1]:
        base_volume = st.number_input("Base hourly volume", 5, 200, 40, key="cc_vol")
    with cols[2]:
        base_aht = st.number_input("Base AHT (sec)", 60, 900, 300, key="cc_aht")
    with cols[3]:
        seed = st.number_input("Seed", 0, 9999, 42, key="cc_seed")
    if st.button("Generate contact-center demand", type="primary", key="gen_cc"):
        rng = np.random.default_rng(int(seed))
        start = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(days=int(days))
        timestamps = pd.date_range(start, periods=int(days) * 24, freq="h")
        hour = timestamps.hour
        dow = timestamps.dayofweek
        hour_factor = np.array(
            [0.2 if h < 7 or h > 20 else (1.3 if 9 <= h <= 16 else 0.8) for h in hour]
        )
        dow_factor = np.array([1.2 if d == 0 else (0.55 if d >= 5 else 1.0) for d in dow])
        volume = np.maximum(
            0,
            (base_volume * hour_factor * dow_factor * rng.normal(1.0, 0.08, len(timestamps))).astype(int),
        )
        aht = np.maximum(30.0, base_aht * rng.normal(1.0, 0.05, len(timestamps)))
        demand_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "volume": volume,
                "average_handle_time": aht,
                "scheduled_agents": int(scheduled_override) if scheduled_override else np.nan,
            }
        )
        st.session_state["staffing_demand"] = demand_df
        st.success(f"Generated {len(demand_df)} hourly intervals")

with tab_sales:
    st.markdown(
        "Use loaded sales models to forecast demand, then map sales → contact volume "
        "(configurable conversion) for staffing."
    )
    if not st.session_state.models_loaded:
        st.info("Load models in the sidebar to enable this path.")
    else:
        fc_days = st.slider("Forecast days", 1, 30, 7, key="sales_fc_days")
        conversion = st.number_input(
            "Contacts per $1000 sales",
            min_value=0.1,
            value=2.0,
            help="volume ≈ predicted_sales / 1000 * this factor",
        )
        aht_sales = st.number_input("Assumed AHT (sec)", 60, 900, 300, key="sales_aht")
        if st.button("Forecast sales → staffing demand", key="sales_to_staff"):
            # sample history then predict
            hist_days = 45
            dates = pd.date_range(end=datetime.now(), periods=hist_days, freq="D")
            sales = 5000 + 200 * np.sin(2 * np.pi * np.arange(hist_days) / 7)
            hist = pd.DataFrame({"date": dates, "store_id": "store_001", "sales": sales})
            results = st.session_state.predictor.predict(
                hist, model_type="ensemble", forecast_days=int(fc_days)
            )
            if not results.get("success"):
                st.error(results.get("error", "Prediction failed"))
            else:
                pred = results["predictions"].copy()
                # predictor returns only future rows in predictions_df in some paths;
                # use predicted_sales column
                if "predicted_sales" not in pred.columns:
                    st.error("Unexpected prediction schema")
                else:
                    volume = np.maximum(
                        0, (pred["predicted_sales"] / 1000.0 * conversion).round().astype(int)
                    )
                    demand_df = pd.DataFrame(
                        {
                            "timestamp": pd.to_datetime(pred["date"]),
                            "volume": volume,
                            "average_handle_time": float(aht_sales),
                            "forecast_lower": pred.get("lower_bound"),
                            "forecast_upper": pred.get("upper_bound"),
                            "forecasted_demand": pred["predicted_sales"],
                        }
                    )
                    st.session_state["staffing_demand"] = demand_df
                    st.session_state["staffing_sales_pred"] = pred
                    st.success("Built staffing demand from sales forecast")

with tab_upload:
    uploaded = st.file_uploader(
        "CSV with columns: timestamp (or date), volume, average_handle_time",
        type=["csv"],
    )
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        if "timestamp" not in raw.columns and "date" in raw.columns:
            raw = raw.rename(columns={"date": "timestamp"})
        need = {"timestamp", "volume", "average_handle_time"}
        if not need.issubset(raw.columns):
            st.error(f"Missing columns: {need - set(raw.columns)}")
        else:
            demand_df = raw.copy()
            demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
            st.session_state["staffing_demand"] = demand_df
            st.success(f"Loaded {len(demand_df)} rows")

demand_df = st.session_state.get("staffing_demand")

if demand_df is None:
    st.info("Generate or upload demand data to build a staffing plan.")
    st.stop()

# Infer interval length from timestamps (hourly vs daily sales forecasts)
if "timestamp" in demand_df.columns and len(demand_df) >= 2:
    deltas = pd.to_datetime(demand_df["timestamp"]).sort_values().diff().dropna()
    if not deltas.empty:
        inferred = int(deltas.median().total_seconds())
        if inferred > 0:
            assumptions = StaffingAssumptions(
                interval_seconds=inferred,
                occupancy_target=assumptions.occupancy_target,
                shrinkage=assumptions.shrinkage,
                service_level_target=assumptions.service_level_target,
                answer_time_seconds=assumptions.answer_time_seconds,
                method=assumptions.method,
                max_agents=assumptions.max_agents,
            )
            calc = StaffingCalculator(assumptions)

# --- Calculate staffing ---
staffed = calc.calculate_frame(demand_df)
if scheduled_override and scheduled_override > 0:
    staffed["scheduled_agents"] = float(scheduled_override)
elif "scheduled_agents" not in staffed.columns:
    staffed["scheduled_agents"] = staffed["required_agents"]
else:
    staffed["scheduled_agents"] = staffed["scheduled_agents"].fillna(staffed["required_agents"])
staffed["staffing_gap"] = staffed["required_agents"] - staffed["scheduled_agents"]

# Prediction intervals on volume (simple ±10% if not provided)
if "forecast_lower" not in staffed.columns:
    staffed["volume_lower"] = (staffed["volume"] * 0.9).clip(lower=0)
    staffed["volume_upper"] = staffed["volume"] * 1.1
else:
    staffed["volume_lower"] = staffed["forecast_lower"]
    staffed["volume_upper"] = staffed["forecast_upper"]

# Metrics row
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total volume", f"{staffed['volume'].sum():,.0f}")
with c2:
    st.metric("Peak required agents", f"{staffed['required_agents'].max():.0f}")
with c3:
    st.metric("Avg required agents", f"{staffed['required_agents'].mean():.1f}")
with c4:
    st.metric("Max staffing gap", f"{staffed['staffing_gap'].max():.0f}")
with c5:
    under = (staffed["staffing_gap"] > 0).sum()
    st.metric("Intervals understaffed", f"{under}")

# Charts
st.subheader("Forecasted demand with intervals")
fig_d = go.Figure()
x = staffed["timestamp"] if "timestamp" in staffed.columns else staffed.index
fig_d.add_trace(go.Scatter(x=x, y=staffed["volume"], name="Volume", line=dict(color="#2E86AB", width=2)))
fig_d.add_trace(
    go.Scatter(x=x, y=staffed["volume_upper"], line=dict(width=0), showlegend=False)
)
fig_d.add_trace(
    go.Scatter(
        x=x,
        y=staffed["volume_lower"],
        fill="tonexty",
        name="Interval",
        line=dict(width=0),
        fillcolor="rgba(46,134,171,0.2)",
    )
)
fig_d.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
st.plotly_chart(fig_d, use_container_width=True)

st.subheader("Required staffing vs scheduled (gap)")
fig_s = go.Figure()
fig_s.add_trace(
    go.Scatter(x=x, y=staffed["required_agents"], name="Required", line=dict(color="#E94F37", width=2))
)
fig_s.add_trace(
    go.Scatter(x=x, y=staffed["scheduled_agents"], name="Scheduled", line=dict(color="#44AF69", width=2))
)
fig_s.add_trace(
    go.Bar(x=x, y=staffed["staffing_gap"], name="Gap (req − sched)", marker_color="#F6AE2D", opacity=0.5)
)
fig_s.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified", barmode="overlay")
st.plotly_chart(fig_s, use_container_width=True)

st.subheader("Plan table")
show_cols = [
    c
    for c in [
        "timestamp",
        "volume",
        "average_handle_time",
        "workload",
        "productive_agents",
        "required_agents",
        "scheduled_agents",
        "staffing_gap",
        "occupancy_used",
        "service_level_estimated",
    ]
    if c in staffed.columns
]
st.dataframe(staffed[show_cols].round(2), use_container_width=True, height=320)

csv = staffed[show_cols].to_csv(index=False)
st.download_button(
    "📥 Download staffing plan (CSV)",
    data=csv,
    file_name=f"staffing_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)
