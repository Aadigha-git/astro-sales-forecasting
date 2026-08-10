"""
Lightweight workforce staffing calculations for operational demand forecasts.

Converts forecasted contact volume and average handle time (AHT) into:
  - offered workload (Erlang / agent-hours)
  - required productive agents (occupancy + optional service-level target)
  - required paid / rostered agents after shrinkage

Does not depend on Airflow, MLflow, or the training DAG — call it from
notebooks, scripts, or inference code when you have volume + AHT forecasts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

Number = Union[int, float, np.ndarray, pd.Series]


@dataclass
class StaffingAssumptions:
    """Configurable WFM assumptions for staffing estimates."""

    # Length of the forecast interval that ``volume`` covers (seconds).
    # Hourly forecasts → 3600; 15-minute forecasts → 900.
    interval_seconds: int = 3600

    # Target productive occupancy (0–1). Workload / agents ≈ occupancy.
    occupancy_target: float = 0.85

    # Non-productive time share (PTO, meetings, breaks, training, etc.).
    # Required rostered = productive / (1 - shrinkage).
    shrinkage: float = 0.30

    # Service-level target: fraction of contacts answered within answer_time_seconds.
    service_level_target: float = 0.80
    answer_time_seconds: float = 20.0

    # "erlang_c" uses Erlang-C for voice-like queues; "occupancy" ignores SL wait target.
    method: str = "erlang_c"

    # Safety cap so pathological inputs cannot loop forever.
    max_agents: int = 5000

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if not 0 < self.occupancy_target <= 1:
            raise ValueError("occupancy_target must be in (0, 1]")
        if not 0 <= self.shrinkage < 1:
            raise ValueError("shrinkage must be in [0, 1)")
        if not 0 < self.service_level_target <= 1:
            raise ValueError("service_level_target must be in (0, 1]")
        if self.answer_time_seconds < 0:
            raise ValueError("answer_time_seconds must be >= 0")
        if self.method not in {"erlang_c", "occupancy"}:
            raise ValueError("method must be 'erlang_c' or 'occupancy'")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StaffingAssumptions":
        """Build from a plain dict / YAML fragment (unknown keys ignored)."""
        valid = {f.name for f in fields(cls)}
        sl = data.get("service_level") or {}
        merged = {k: v for k, v in data.items() if k in valid}
        if "target" in sl and "service_level_target" not in merged:
            merged["service_level_target"] = sl["target"]
        if "answer_time_seconds" in sl and "answer_time_seconds" not in data:
            merged["answer_time_seconds"] = sl["answer_time_seconds"]
        return cls(**merged)

    @classmethod
    def from_yaml(
        cls,
        config_path: Union[str, Path] = "/usr/local/airflow/include/config/ml_config.yaml",
    ) -> "StaffingAssumptions":
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found at {path}; using defaults")
            return cls()
        with path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        return cls.from_mapping(cfg.get("workforce", {}) or {})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_workload(
    volume: Number,
    average_handle_time: Number,
    interval_seconds: int = 3600,
) -> np.ndarray:
    """
    Workload in agent-intervals (Erlangs for the period).

    workload = (volume * AHT_seconds) / interval_seconds

    Example: 120 calls × 300s AHT / 3600s = 10.0 agent-hours of work.
    """
    vol = np.asarray(volume, dtype=float)
    aht = np.asarray(average_handle_time, dtype=float)
    if np.any(vol < 0) or np.any(aht < 0):
        raise ValueError("volume and average_handle_time must be non-negative")
    return (vol * aht) / float(interval_seconds)


def _erlang_c_probability(agents: int, traffic_intensity: float) -> float:
    """
    Probability that an arriving contact waits (Erlang C).

    ``traffic_intensity`` (A) is offered load in Erlangs; ``agents`` (N) must be > A
    for a stable queue. Returns 1.0 if N <= A.
    """
    if agents <= 0:
        return 1.0
    a = float(traffic_intensity)
    if a < 0:
        raise ValueError("traffic_intensity must be >= 0")
    if agents <= a:
        return 1.0

    # Recurrence form avoids huge factorials for large N.
    # Pw = (A^N / N!) / (sum_{k=0}^{N-1} A^k/k! + A^N/N!)
    # Compute via iterative product for numerical stability.
    n = agents
    term = 1.0  # A^0 / 0!
    total = term
    for k in range(1, n + 1):
        term *= a / k
        total += term
    last = term  # A^N / N!
    return last / total if total > 0 else 1.0


def service_level_achieved(
    agents: int,
    traffic_intensity: float,
    average_handle_time: float,
    answer_time_seconds: float,
) -> float:
    """
    Estimated service level: P(wait <= answer_time).

    SL = 1 - P(wait>0) * exp(-(N - A) * T / AHT)
    """
    if agents <= 0:
        return 0.0
    a = float(traffic_intensity)
    aht = max(float(average_handle_time), 1e-6)
    pw = _erlang_c_probability(agents, a)
    delay_factor = math.exp(-(agents - a) * float(answer_time_seconds) / aht)
    return float(max(0.0, min(1.0, 1.0 - pw * delay_factor)))


def agents_for_service_level(
    traffic_intensity: float,
    average_handle_time: float,
    service_level_target: float = 0.80,
    answer_time_seconds: float = 20.0,
    max_agents: int = 5000,
) -> int:
    """Smallest agent count meeting the service-level target (Erlang C)."""
    a = float(traffic_intensity)
    if a <= 0:
        return 0

    # Need strictly more agents than offered load for a stable queue.
    n = max(1, math.ceil(a) + 1)
    aht = float(average_handle_time)

    while n <= max_agents:
        sl = service_level_achieved(n, a, aht, answer_time_seconds)
        if sl >= service_level_target:
            return n
        n += 1

    logger.warning(
        f"Could not meet SL={service_level_target:.0%} within max_agents={max_agents}; "
        f"returning cap (A={a:.2f})"
    )
    return max_agents


def apply_shrinkage(productive_agents: Number, shrinkage: float) -> np.ndarray:
    """Rostered / paid headcount needed given shrinkage."""
    if not 0 <= shrinkage < 1:
        raise ValueError("shrinkage must be in [0, 1)")
    prod = np.asarray(productive_agents, dtype=float)
    return prod / (1.0 - float(shrinkage))


@dataclass
class StaffingResult:
    """Scalar staffing estimate for one interval."""

    volume: float
    average_handle_time: float
    workload: float
    productive_agents: float
    required_agents: float
    occupancy_used: float
    service_level_target: float
    service_level_estimated: Optional[float]
    shrinkage: float
    method: str
    interval_seconds: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StaffingCalculator:
    """
    Convert forecasted volume + AHT into workload and staffing requirements.

    Example
    -------
    >>> calc = StaffingCalculator(StaffingAssumptions(occupancy_target=0.85, shrinkage=0.3))
    >>> calc.calculate(volume=120, average_handle_time=300)
    """

    def __init__(self, assumptions: Optional[StaffingAssumptions] = None):
        self.assumptions = assumptions or StaffingAssumptions()

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path] = "/usr/local/airflow/include/config/ml_config.yaml",
    ) -> "StaffingCalculator":
        return cls(StaffingAssumptions.from_yaml(config_path))

    def calculate(
        self,
        volume: float,
        average_handle_time: float,
        assumptions: Optional[StaffingAssumptions] = None,
    ) -> StaffingResult:
        """Staffing estimate for a single interval."""
        cfg = assumptions or self.assumptions
        vol = float(max(0.0, volume))
        aht = float(max(0.0, average_handle_time))

        workload = float(compute_workload(vol, aht, cfg.interval_seconds))

        if workload <= 0:
            productive = 0.0
            sl_est = 1.0 if vol == 0 else 0.0
        elif cfg.method == "occupancy":
            productive = math.ceil(workload / cfg.occupancy_target)
            sl_est = None
        else:
            # Erlang-C agents for SL, then enforce occupancy floor
            sl_agents = agents_for_service_level(
                traffic_intensity=workload,
                average_handle_time=aht if aht > 0 else 1.0,
                service_level_target=cfg.service_level_target,
                answer_time_seconds=cfg.answer_time_seconds,
                max_agents=cfg.max_agents,
            )
            occ_agents = math.ceil(workload / cfg.occupancy_target)
            productive = float(max(sl_agents, occ_agents))
            sl_est = service_level_achieved(
                int(productive),
                workload,
                aht if aht > 0 else 1.0,
                cfg.answer_time_seconds,
            )

        required = float(math.ceil(float(apply_shrinkage(productive, cfg.shrinkage))))
        occ_used = float(workload / productive) if productive > 0 else 0.0

        return StaffingResult(
            volume=vol,
            average_handle_time=aht,
            workload=workload,
            productive_agents=float(productive),
            required_agents=required,
            occupancy_used=occ_used,
            service_level_target=cfg.service_level_target,
            service_level_estimated=sl_est,
            shrinkage=cfg.shrinkage,
            method=cfg.method,
            interval_seconds=cfg.interval_seconds,
        )

    def calculate_frame(
        self,
        df: pd.DataFrame,
        volume_col: str = "volume",
        aht_col: str = "average_handle_time",
        assumptions: Optional[StaffingAssumptions] = None,
    ) -> pd.DataFrame:
        """
        Vectorized-friendly row-wise staffing for a forecast DataFrame.

        Expects columns for volume and AHT (defaults match contact-center generator).
        Appends: workload, productive_agents, required_agents, occupancy_used,
        service_level_estimated.
        """
        if volume_col not in df.columns:
            raise KeyError(f"Missing volume column: {volume_col}")
        if aht_col not in df.columns:
            raise KeyError(f"Missing AHT column: {aht_col}")

        cfg = assumptions or self.assumptions
        out = df.copy()
        results = [
            self.calculate(
                volume=float(row[volume_col]),
                average_handle_time=float(row[aht_col]),
                assumptions=cfg,
            )
            for _, row in out.iterrows()
        ]
        out["workload"] = [r.workload for r in results]
        out["productive_agents"] = [r.productive_agents for r in results]
        out["required_agents"] = [r.required_agents for r in results]
        out["occupancy_used"] = [r.occupancy_used for r in results]
        out["service_level_estimated"] = [r.service_level_estimated for r in results]
        out["staffing_method"] = cfg.method
        out["shrinkage"] = cfg.shrinkage
        out["occupancy_target"] = cfg.occupancy_target
        return out

    def summarize_by(
        self,
        df: pd.DataFrame,
        group_cols: Optional[list] = None,
        volume_col: str = "volume",
        aht_col: str = "average_handle_time",
    ) -> pd.DataFrame:
        """
        Aggregate volume-weighted AHT by group, then compute staffing once per group.

        Useful for daily / business-unit / channel rollups from hourly forecasts.
        """
        group_cols = group_cols or []
        if group_cols:
            rows = []
            for keys, g in df.groupby(group_cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                vol = float(g[volume_col].sum())
                aht = (
                    float(np.average(g[aht_col], weights=g[volume_col]))
                    if vol > 0
                    else float(g[aht_col].mean())
                )
                row = dict(zip(group_cols, keys))
                row[volume_col] = vol
                row[aht_col] = aht
                rows.append(row)
            grouped = pd.DataFrame(rows)
        else:
            vol = float(df[volume_col].sum())
            aht = (
                float(np.average(df[aht_col], weights=df[volume_col]))
                if vol > 0
                else float(df[aht_col].mean())
            )
            grouped = pd.DataFrame([{volume_col: vol, aht_col: aht}])

        return self.calculate_frame(grouped, volume_col=volume_col, aht_col=aht_col)
