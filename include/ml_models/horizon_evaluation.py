"""
Multi-horizon forecast evaluation (1 / 3 / 7 / 14 day).

Fits models at rolling origins on a daily series, scores cumulative
forecast windows at each horizon, and aggregates accuracy + bias by horizon.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ml_models.rolling_origin_cv import (
    RollingOriginEvaluator,
    aggregate_fold_metrics,
    calculate_fold_metrics,
    generate_rolling_origin_folds,
)

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = (1, 3, 7, 14)


def aggregate_to_daily_series(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "sales",
) -> pd.DataFrame:
    """
    Collapse panel / multi-entity rows to one observation per calendar day.

    Adds calendar features for tabular models (no lag features — avoids leakage
    when slicing train/test after the fact).
    """
    data = df[[date_col, target_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    daily = (
        data.groupby(date_col, as_index=False)[target_col]
        .sum()
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    ts = daily[date_col]
    daily["dayofweek"] = ts.dt.dayofweek
    daily["month"] = ts.dt.month
    daily["day"] = ts.dt.day
    daily["weekofyear"] = ts.dt.isocalendar().week.astype(int)
    daily["quarter"] = ts.dt.quarter
    daily["is_weekend"] = (daily["dayofweek"] >= 5).astype(int)
    return daily


class MultiHorizonEvaluator(RollingOriginEvaluator):
    """
    Evaluate forecast accuracy and bias at multiple horizons.

    For each rolling origin, forecasts ``max(horizons)`` steps ahead once, then
    scores the prefix of length h for each requested horizon.
    """

    def evaluate_horizons(
        self,
        df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        n_splits: int = 5,
        step: Optional[int] = None,
        min_train_size: Optional[int] = None,
        model_names: Optional[Sequence[str]] = None,
        aggregate_daily: bool = True,
    ) -> Dict[str, Any]:
        horizons = sorted({int(h) for h in horizons if int(h) > 0})
        if not horizons:
            raise ValueError("At least one positive horizon is required")
        max_horizon = max(horizons)

        if aggregate_daily:
            data = aggregate_to_daily_series(df, date_col=date_col, target_col=target_col)
        else:
            data = df.sort_values(date_col).dropna(subset=[target_col]).reset_index(drop=True)

        n = len(data)
        step = int(step) if step is not None else max_horizon

        folds = generate_rolling_origin_folds(
            n_samples=n,
            n_splits=n_splits,
            horizon=max_horizon,
            min_train_size=min_train_size,
            step=step,
        )
        logger.info(
            f"Multi-horizon eval: horizons={horizons}, folds={len(folds)}, "
            f"daily_points={n}, max_horizon={max_horizon}"
        )

        candidates = list(
            model_names
            or (
                "xgboost",
                "lightgbm",
                "seasonal_naive",
                "holt_winters",
                "sarimax",
                "prophet",
            )
        )
        active_models = [m for m in candidates if self._is_enabled(m)]

        # model -> horizon -> list of fold metrics
        store: Dict[str, Dict[int, List[Dict[str, float]]]] = {
            m: {h: [] for h in horizons} for m in active_models
        }
        fold_details: Dict[str, Dict[int, List[Dict[str, Any]]]] = {
            m: {h: [] for h in horizons} for m in active_models
        }

        for fold in folds:
            fold_train = data.iloc[: fold.train_end].copy()
            fold_test = data.iloc[fold.test_start : fold.test_end].copy()
            y_true_full = fold_test[target_col].astype(float).values

            for model_name in active_models:
                try:
                    y_pred_full = self._forecast_fold(
                        model_name, fold_train, fold_test, target_col, date_col
                    )
                    if len(y_pred_full) < max_horizon:
                        raise ValueError(
                            f"Forecast length {len(y_pred_full)} < max horizon {max_horizon}"
                        )
                    for h in horizons:
                        y_true_h = y_true_full[:h]
                        y_pred_h = np.asarray(y_pred_full[:h], dtype=float)
                        metrics = self.metrics_fn(y_true_h, y_pred_h)
                        store[model_name][h].append(metrics)
                        fold_details[model_name][h].append(
                            {
                                "fold_id": fold.fold_id,
                                "train_end": fold.train_end,
                                "horizon": h,
                                "metrics": metrics,
                            }
                        )
                except Exception as e:
                    logger.warning(
                        f"Multi-horizon fold {fold.fold_id} failed for {model_name}: {e}"
                    )

        models_out: Dict[str, Any] = {}
        accuracy_rows: List[Dict[str, Any]] = []

        for model_name in active_models:
            by_horizon: Dict[str, Any] = {}
            horizon_summary: List[Dict[str, Any]] = []
            for h in horizons:
                aggregated = aggregate_fold_metrics(store[model_name][h])
                by_horizon[str(h)] = {
                    "folds": fold_details[model_name][h],
                    "aggregated": aggregated,
                }
                summary_row = {
                    "model": model_name,
                    "horizon": h,
                    "n_folds": int(aggregated.get("n_folds", 0)),
                    "rmse": aggregated.get("rmse_mean", float("nan")),
                    "mae": aggregated.get("mae_mean", float("nan")),
                    "wape": aggregated.get("wape_mean", float("nan")),
                    "mape": aggregated.get("mape_mean", float("nan")),
                    "bias": aggregated.get("bias_mean", float("nan")),
                    "rmse_std": aggregated.get("rmse_std", float("nan")),
                    "mae_std": aggregated.get("mae_std", float("nan")),
                    "wape_std": aggregated.get("wape_std", float("nan")),
                    "bias_std": aggregated.get("bias_std", float("nan")),
                }
                horizon_summary.append(summary_row)
                accuracy_rows.append(summary_row)
                if aggregated:
                    logger.info(
                        f"Horizon {h}d {model_name}: "
                        f"RMSE={summary_row['rmse']:.4f}, "
                        f"MAE={summary_row['mae']:.4f}, "
                        f"WAPE={summary_row['wape']:.4f}, "
                        f"bias={summary_row['bias']:.4f}"
                    )

            models_out[model_name] = {
                "by_horizon": by_horizon,
                "horizon_summary": horizon_summary,
            }

        return {
            "config": {
                "horizons": list(horizons),
                "n_splits_requested": n_splits,
                "n_folds": len(folds),
                "max_horizon": max_horizon,
                "step": step,
                "min_train_size": min_train_size,
                "n_samples": n,
                "aggregate_daily": aggregate_daily,
            },
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "train_end": f.train_end,
                    "test_start": f.test_start,
                    "test_end": f.test_end,
                }
                for f in folds
            ],
            "models": models_out,
            "accuracy_by_horizon": accuracy_rows,
        }


def horizon_summary_frame(horizon_results: Dict[str, Any]) -> pd.DataFrame:
    """Convenience DataFrame of model × horizon accuracy / bias."""
    rows = horizon_results.get("accuracy_by_horizon") or []
    if not rows:
        return pd.DataFrame(
            columns=["model", "horizon", "rmse", "mae", "wape", "bias", "n_folds"]
        )
    return pd.DataFrame(rows).sort_values(["model", "horizon"]).reset_index(drop=True)
