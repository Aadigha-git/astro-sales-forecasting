"""
Classical time-series forecasting models for the sales forecast pipeline.

Provides sklearn-like fit/predict wrappers around Seasonal Naive,
Holt-Winters (Exponential Smoothing), and SARIMAX so they plug into
ModelTrainer comparison and ensemble evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]


def _as_1d_array(y: ArrayLike) -> np.ndarray:
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Input series is empty")
    if np.isnan(arr).any():
        raise ValueError("Input series contains NaN values")
    return arr


class SeasonalNaiveModel:
    """Forecast by repeating the last full seasonal cycle."""

    def __init__(self, season_length: int = 7):
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = int(season_length)
        self.history_: Optional[np.ndarray] = None
        self.seasonal_pattern_: Optional[np.ndarray] = None

    def fit(self, y: ArrayLike) -> "SeasonalNaiveModel":
        history = _as_1d_array(y)
        if len(history) < self.season_length:
            raise ValueError(
                f"Need at least {self.season_length} observations for Seasonal Naive, "
                f"got {len(history)}"
            )
        self.history_ = history
        self.seasonal_pattern_ = history[-self.season_length :].copy()
        return self

    def predict(self, steps: int) -> np.ndarray:
        self._ensure_fitted()
        if steps < 1:
            return np.array([], dtype=float)
        reps = int(np.ceil(steps / self.season_length))
        return np.tile(self.seasonal_pattern_, reps)[:steps]

    def forecast(self, steps: int) -> np.ndarray:
        return self.predict(steps)

    def _ensure_fitted(self) -> None:
        if self.seasonal_pattern_ is None:
            raise RuntimeError("SeasonalNaiveModel must be fitted before predict")


class HoltWintersModel:
    """Holt-Winters / Exponential Smoothing wrapper."""

    def __init__(
        self,
        seasonal: str = "add",
        seasonal_periods: int = 7,
        trend: Optional[str] = "add",
        damped_trend: bool = False,
        use_boxcox: bool = False,
        initialization_method: str = "estimated",
    ):
        self.seasonal = seasonal
        self.seasonal_periods = int(seasonal_periods)
        self.trend = trend
        self.damped_trend = damped_trend
        self.use_boxcox = use_boxcox
        self.initialization_method = initialization_method
        self.model_ = None
        self.fitted_ = None
        self.history_: Optional[np.ndarray] = None

    def fit(self, y: ArrayLike) -> "HoltWintersModel":
        history = _as_1d_array(y)
        min_obs = max(2 * self.seasonal_periods, self.seasonal_periods + 2)
        if len(history) < min_obs:
            raise ValueError(
                f"Need at least {min_obs} observations for Holt-Winters, got {len(history)}"
            )

        # Box-Cox requires strictly positive data
        use_boxcox = self.use_boxcox and np.all(history > 0)

        self.history_ = history
        self.model_ = ExponentialSmoothing(
            history,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            use_boxcox=use_boxcox,
            initialization_method=self.initialization_method,
        )
        self.fitted_ = self.model_.fit(optimized=True)
        return self

    def predict(self, steps: int) -> np.ndarray:
        self._ensure_fitted()
        if steps < 1:
            return np.array([], dtype=float)
        return np.asarray(self.fitted_.forecast(steps), dtype=float)

    def forecast(self, steps: int) -> np.ndarray:
        return self.predict(steps)

    def _ensure_fitted(self) -> None:
        if self.fitted_ is None:
            raise RuntimeError("HoltWintersModel must be fitted before predict")


class SARIMAXModel:
    """SARIMAX wrapper with optional exogenous regressors."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        trend: Optional[str] = None,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
        maxiter: int = 50,
    ):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.maxiter = maxiter
        self.model_ = None
        self.fitted_ = None
        self.history_: Optional[np.ndarray] = None
        self.exog_names_: Optional[list] = None

    def fit(
        self, y: ArrayLike, exog: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> "SARIMAXModel":
        history = _as_1d_array(y)
        season_len = self.seasonal_order[3] if len(self.seasonal_order) == 4 else 0
        min_obs = max(2 * max(season_len, 1), 10)
        if len(history) < min_obs:
            raise ValueError(
                f"Need at least {min_obs} observations for SARIMAX, got {len(history)}"
            )

        exog_arr = None
        if exog is not None:
            if isinstance(exog, pd.DataFrame):
                self.exog_names_ = list(exog.columns)
                exog_arr = exog.to_numpy(dtype=float)
            else:
                exog_arr = np.asarray(exog, dtype=float)
                if exog_arr.ndim == 1:
                    exog_arr = exog_arr.reshape(-1, 1)
            if len(exog_arr) != len(history):
                raise ValueError("exog length must match y length")

        self.history_ = history
        self.model_ = SARIMAX(
            history,
            exog=exog_arr,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self.fitted_ = self.model_.fit(disp=False, maxiter=self.maxiter)
        return self

    def predict(
        self,
        steps: int,
        exog: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> np.ndarray:
        self._ensure_fitted()
        if steps < 1:
            return np.array([], dtype=float)

        exog_arr = None
        if exog is not None:
            if isinstance(exog, pd.DataFrame):
                if self.exog_names_ is not None:
                    exog = exog[self.exog_names_]
                exog_arr = exog.to_numpy(dtype=float)
            else:
                exog_arr = np.asarray(exog, dtype=float)
                if exog_arr.ndim == 1:
                    exog_arr = exog_arr.reshape(-1, 1)
            if len(exog_arr) != steps:
                raise ValueError("exog forecast length must equal steps")

        forecast = self.fitted_.get_forecast(steps=steps, exog=exog_arr)
        return np.asarray(forecast.predicted_mean, dtype=float)

    def forecast(
        self,
        steps: int,
        exog: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> np.ndarray:
        return self.predict(steps, exog=exog)

    def _ensure_fitted(self) -> None:
        if self.fitted_ is None:
            raise RuntimeError("SARIMAXModel must be fitted before predict")


def build_model_from_config(model_name: str, params: Dict[str, Any]):
    """Factory used by ModelTrainer for config-driven construction."""
    params = dict(params or {})
    if model_name == "seasonal_naive":
        return SeasonalNaiveModel(**params)
    if model_name in ("holt_winters", "exponential_smoothing"):
        return HoltWintersModel(**params)
    if model_name == "sarimax":
        order = params.pop("order", (1, 1, 1))
        seasonal_order = params.pop("seasonal_order", (1, 1, 1, 7))
        return SARIMAXModel(order=tuple(order), seasonal_order=tuple(seasonal_order), **params)
    raise ValueError(f"Unknown classical model: {model_name}")
