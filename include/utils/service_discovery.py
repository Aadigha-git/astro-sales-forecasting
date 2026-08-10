"""
Simple service discovery for MLflow and MinIO endpoints.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from typing import List, Optional

logger = logging.getLogger(__name__)


def _in_container() -> bool:
    return os.path.exists("/.dockerenv") or bool(os.environ.get("AIRFLOW__CORE__EXECUTOR"))


def _probe(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.getcode() < 300
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.debug("Probe failed for %s: %s", url, exc)
        return False


def _unique(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def get_mlflow_endpoint() -> str:
    """Return a reachable MLflow tracking URI."""
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if _in_container():
        candidates = [
            env_uri,
            "http://mlflow:5001",
            "http://host.docker.internal:5001",
            "http://172.17.0.1:5001",
        ]
    else:
        candidates = [
            env_uri,
            "http://localhost:5001",
            "http://127.0.0.1:5001",
            "http://host.docker.internal:5001",
        ]

    for endpoint in _unique(candidates):
        if _probe(f"{endpoint.rstrip('/')}/health"):
            logger.info("MLflow is accessible at: %s", endpoint)
            return endpoint

    default = "http://mlflow:5001" if _in_container() else "http://localhost:5001"
    logger.warning("Could not connect to MLflow, using default: %s", default)
    return default


def get_minio_endpoint() -> str:
    """Return a reachable MinIO S3 endpoint URL."""
    env_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    if _in_container():
        candidates = [
            env_url,
            "http://minio:9000",
            "http://host.docker.internal:9000",
            "http://172.17.0.1:9000",
        ]
    else:
        candidates = [
            env_url,
            "http://localhost:9000",
            "http://127.0.0.1:9000",
            "http://host.docker.internal:9000",
        ]

    for endpoint in _unique(candidates):
        if _probe(f"{endpoint.rstrip('/')}/minio/health/live"):
            logger.info("MinIO is accessible at: %s", endpoint)
            return endpoint

    default = "http://minio:9000" if _in_container() else "http://localhost:9000"
    logger.warning("Could not connect to MinIO, using default: %s", default)
    return default


def get_mlflow_uri() -> str:
    """Get MLflow URI (backward compatibility)."""
    return get_mlflow_endpoint()


def get_minio_url() -> str:
    """Get MinIO URL (backward compatibility)."""
    return get_minio_endpoint()
