"""
Rolling-origin (expanding-window) time-series cross-validation.

Evaluates forecasting models across multiple historical forecast origins
and aggregates fold-level metrics. Designed to run on train+validation
data only so the final holdout test set stays untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml_models.classical_ts_models import build_model_from_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RollingOriginFold:
    """One rolling-origin forecast window."""

    fold_id: int
    train_end: int  # exclusive index into chronologically sorted frame
    test_start: int
    test_end: int  # exclusive

    @property
    def horizon(self) -> int:
        return self.test_end - self.test_start

    @property
    def train_size(self) -> int:
        return self.train_end


def generate_rolling_origin_folds(
    n_samples: int,
    n_splits: int = 5,
    horizon: int = 30,
    min_train_size: Optional[int] = None,
    step: Optional[int] = None,
) -> List[RollingOriginFold]:
    """
    Build expanding-window forecast origins.

    For each fold i:
      train = [0, origin)
      test  = [origin, origin + horizon)
    Origins advance by ``step`` (default = horizon) toward the end of the series.
    """
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for rolling-origin CV")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    step = int(step) if step is not None else int(horizon)
    if step < 1:
        raise ValueError("step must be >= 1")

    if min_train_size is None:
        # Keep enough history for seasonality / GBDT while leaving room for folds
        reserved = horizon + (n_splits - 1) * step
        min_train_size = max(2 * horizon, n_samples - reserved, int(n_samples * 0.4))
    min_train_size = int(min_train_size)

    max_origin = n_samples - horizon
    if max_origin < min_train_size:
        raise ValueError(
            f"Not enough samples for rolling-origin CV: "
            f"n={n_samples}, horizon={horizon}, min_train_size={min_train_size}"
        )

    folds: List[RollingOriginFold] = []
    for i in range(n_splits):
        origin = max_origin - (n_splits - 1 - i) * step
        if origin < min_train_size:
            continue
        folds.append(
            RollingOriginFold(
                fold_id=len(folds),
                train_end=origin,
                test_start=origin,
                test_end=origin + horizon,
            )
        )

    if not folds:
        raise ValueError(
            "Could not construct any rolling-origin folds with the given parameters"
        )
    return folds


def calculate_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    abs_err = np.abs(y_true - y_pred)
    actual_sum = np.sum(np.abs(y_true))
    wape = float(np.sum(abs_err) / actual_sum * 100) if actual_sum > 0 else float("nan")
    bias = float(np.mean(y_pred - y_true))

    mape_denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / mape_denom)) * 100)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape,
        "wape": wape,
        "bias": bias,
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def aggregate_fold_metrics(
    fold_metrics: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """Mean / std across folds for each metric."""
    if not fold_metrics:
        return {}
    keys = sorted({k for m in fold_metrics for k in m})
    aggregated: Dict[str, float] = {"n_folds": float(len(fold_metrics))}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics if key in m], dtype=float)
        if values.size == 0:
            continue
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values, ddof=0))
        aggregated[f"{key}_min"] = float(np.min(values))
        aggregated[f"{key}_max"] = float(np.max(values))
    return aggregated


class RollingOriginEvaluator:
    """
    Run rolling-origin CV for tabular + classical forecasting models.

    Does not mutate the caller's fitted production models / scalers.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        metrics_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None,
    ):
        self.model_config = model_config
        self.metrics_fn = metrics_fn or calculate_fold_metrics

    def _is_enabled(self, model_name: str, default: bool = False) -> bool:
        cfg = self.model_config.get(model_name, {})
        if model_name in ("xgboost", "lightgbm"):
            return True
        return bool(cfg.get("enabled", default))

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        n_splits: int = 5,
        horizon: int = 30,
        min_train_size: Optional[int] = None,
        step: Optional[int] = None,
        model_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate models over rolling-origin folds on ``df`` (chronological).

        Returns per-model fold details + aggregated metrics, plus fold metadata.
        """
        data = df.sort_values(date_col).reset_index(drop=True)
        data = data.dropna(subset=[target_col]).reset_index(drop=True)
        n = len(data)

        folds = generate_rolling_origin_folds(
            n_samples=n,
            n_splits=n_splits,
            horizon=horizon,
            min_train_size=min_train_size,
            step=step,
        )
        logger.info(
            f"Rolling-origin CV: {len(folds)} folds, horizon={horizon}, "
            f"n_samples={n}, first_train_end={folds[0].train_end}, "
            f"last_train_end={folds[-1].train_end}"
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

        per_model: Dict[str, Dict[str, Any]] = {
            name: {"folds": [], "fold_metrics": []} for name in active_models
        }

        for fold in folds:
            fold_train = data.iloc[: fold.train_end].copy()
            fold_test = data.iloc[fold.test_start : fold.test_end].copy()
            y_true = fold_test[target_col].astype(float).values

            logger.info(
                f"CV fold {fold.fold_id}: train={len(fold_train)}, "
                f"test={len(fold_test)}, origin_idx={fold.train_end}"
            )

            for model_name in active_models:
                try:
                    y_pred = self._forecast_fold(
                        model_name, fold_train, fold_test, target_col, date_col
                    )
                    if len(y_pred) != len(y_true):
                        raise ValueError(
                            f"Prediction length {len(y_pred)} != holdout {len(y_true)}"
                        )
                    metrics = self.metrics_fn(y_true, y_pred)
                    fold_record = {
                        "fold_id": fold.fold_id,
                        "train_end": fold.train_end,
                        "test_start": fold.test_start,
                        "test_end": fold.test_end,
                        "horizon": fold.horizon,
                        "train_size": fold.train_size,
                        "metrics": metrics,
                    }
                    per_model[model_name]["folds"].append(fold_record)
                    per_model[model_name]["fold_metrics"].append(metrics)
                except Exception as e:
                    logger.warning(
                        f"Rolling-origin CV fold {fold.fold_id} failed for "
                        f"{model_name}: {e}"
                    )

        results: Dict[str, Any] = {
            "config": {
                "n_splits_requested": n_splits,
                "n_folds": len(folds),
                "horizon": horizon,
                "step": step if step is not None else horizon,
                "min_train_size": min_train_size,
                "n_samples": n,
            },
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "train_end": f.train_end,
                    "test_start": f.test_start,
                    "test_end": f.test_end,
                    "horizon": f.horizon,
                }
                for f in folds
            ],
            "models": {},
        }

        for model_name, payload in per_model.items():
            aggregated = aggregate_fold_metrics(payload["fold_metrics"])
            results["models"][model_name] = {
                "folds": payload["folds"],
                "aggregated": aggregated,
            }
            if aggregated:
                logger.info(
                    f"CV {model_name}: RMSE={aggregated.get('rmse_mean', float('nan')):.4f} "
                    f"± {aggregated.get('rmse_std', float('nan')):.4f} "
                    f"over {int(aggregated.get('n_folds', 0))} folds"
                )

        return results

    def _forecast_fold(
        self,
        model_name: str,
        fold_train: pd.DataFrame,
        fold_test: pd.DataFrame,
        target_col: str,
        date_col: str,
    ) -> np.ndarray:
        if model_name in ("xgboost", "lightgbm"):
            return self._forecast_tabular(
                model_name, fold_train, fold_test, target_col, date_col
            )
        if model_name in ("seasonal_naive", "holt_winters", "sarimax"):
            return self._forecast_classical(
                model_name, fold_train, fold_test, target_col
            )
        if model_name == "prophet":
            return self._forecast_prophet(fold_train, fold_test, target_col, date_col)
        raise ValueError(f"Unsupported model for rolling-origin CV: {model_name}")

    def _forecast_tabular(
        self,
        model_name: str,
        fold_train: pd.DataFrame,
        fold_test: pd.DataFrame,
        target_col: str,
        date_col: str,
    ) -> np.ndarray:
        exclude = [date_col, target_col]
        feature_cols = [c for c in fold_train.columns if c not in exclude]
        if not feature_cols:
            raise ValueError("No feature columns available for tabular CV fold")

        X_train = fold_train[feature_cols].copy()
        X_test = fold_test[feature_cols].copy()
        y_all = fold_train[target_col].astype(float).values

        # Local encoders / scaler — do not touch production artifacts
        for col in X_train.select_dtypes(include=["object"]).columns:
            enc = LabelEncoder()
            X_train[col] = enc.fit_transform(X_train[col].astype(str))
            # Unseen labels → most frequent class index 0 fallback
            test_vals = X_test[col].astype(str)
            known = set(enc.classes_)
            X_test[col] = [
                enc.transform([v])[0] if v in known else 0 for v in test_vals
            ]

        # Small trailing slice of fold train for early stopping
        val_n = max(1, int(len(X_train) * 0.1))
        if len(X_train) > val_n + 5:
            X_fit, X_es = X_train.iloc[:-val_n], X_train.iloc[-val_n:]
            y_fit, y_es = y_all[:-val_n], y_all[-val_n:]
        else:
            X_fit, X_es, y_fit, y_es = X_train, X_train, y_all, y_all

        scaler = StandardScaler()
        X_fit_s = scaler.fit_transform(X_fit)
        X_es_s = scaler.transform(X_es)
        X_test_s = scaler.transform(X_test)

        if model_name == "xgboost":
            import xgboost as xgb

            params = dict(self.model_config.get("xgboost", {}).get("params", {}))
            params.setdefault("random_state", 42)
            params["early_stopping_rounds"] = 25
            model = xgb.XGBRegressor(**params)
            model.fit(X_fit_s, y_fit, eval_set=[(X_es_s, y_es)], verbose=False)
        else:
            import lightgbm as lgb

            params = dict(self.model_config.get("lightgbm", {}).get("params", {}))
            params.setdefault("random_state", 42)
            params.setdefault("verbosity", -1)
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_fit_s,
                y_fit,
                eval_set=[(X_es_s, y_es)],
                callbacks=[lgb.early_stopping(25), lgb.log_evaluation(0)],
            )

        return np.asarray(model.predict(X_test_s), dtype=float)

    def _forecast_classical(
        self,
        model_name: str,
        fold_train: pd.DataFrame,
        fold_test: pd.DataFrame,
        target_col: str,
    ) -> np.ndarray:
        y_history = fold_train[target_col].astype(float).values
        params = dict(self.model_config.get(model_name, {}).get("params", {}))
        model = build_model_from_config(model_name, params)
        model.fit(y_history)
        return np.asarray(model.predict(len(fold_test)), dtype=float)

    def _forecast_prophet(
        self,
        fold_train: pd.DataFrame,
        fold_test: pd.DataFrame,
        target_col: str,
        date_col: str,
    ) -> np.ndarray:
        from prophet import Prophet

        params = dict(self.model_config.get("prophet", {}).get("params", {}))
        params.update(
            {
                "mcmc_samples": 0,
                "uncertainty_samples": 50,
            }
        )
        train = (
            fold_train[[date_col, target_col]]
            .rename(columns={date_col: "ds", target_col: "y"})
            .dropna()
            .sort_values("ds")
        )
        model = Prophet(**{k: v for k, v in params.items() if k != "enabled"})
        model.fit(train)
        future = fold_test[[date_col]].rename(columns={date_col: "ds"})
        return np.asarray(model.predict(future)["yhat"].values, dtype=float)
