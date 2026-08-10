"""Shared path bootstrap so Streamlit pages can import include/ modules."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_include_on_path() -> Path:
    """
    Make ``include/`` importable as ``workforce``, ``ml_models``, etc.

    Prefer INCLUDE_PATH / PYTHONPATH from compose; fall back to repo layout.
    """
    candidates = []
    env_include = os.getenv("INCLUDE_PATH")
    if env_include:
        candidates.append(Path(env_include))
    candidates.extend(
        [
            Path("/usr/local/airflow/include"),
            Path(__file__).resolve().parents[2] / "include",
            Path(__file__).resolve().parents[1].parent / "include",
        ]
    )
    for path in candidates:
        if path.is_dir():
            resolved = str(path.resolve())
            # Append (do not insert at 0) so a UI package named ``utils``
            # under /app is not shadowed by include/utils.
            if resolved not in sys.path:
                sys.path.append(resolved)
            return path
    return Path("/usr/local/airflow/include")


def ml_config_path() -> str:
    return os.getenv(
        "ML_CONFIG_PATH",
        "/usr/local/airflow/include/config/ml_config.yaml",
    )


ensure_include_on_path()
