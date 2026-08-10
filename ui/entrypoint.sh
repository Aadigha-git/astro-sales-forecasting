#!/bin/bash
set -euo pipefail

# Dependencies are baked into the image (ui/Dockerfile.streamlit).
# Code is volume-mounted; do not apt/pip install on every start.
# /app must come before include/ so ui.utils is not shadowed by include.utils.
INCLUDE="${INCLUDE_PATH:-/usr/local/airflow/include}"
export PYTHONPATH="/app:${INCLUDE}${PYTHONPATH:+:${PYTHONPATH}}"

exec streamlit run inference_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --browser.gatherUsageStats false
