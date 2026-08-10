"""
Automated forecast quality checks.

Validates forecast frames / prediction arrays for:
  - missing timestamps (gaps vs expected frequency)
  - duplicate timestamps
  - negative volumes (actuals and/or predictions)
  - missing predictions (nulls, length mismatch, non-finite)
  - excessive forecast bias
  - abnormal forecast errors (RMSE / MAE / WAPE thresholds)

Produces a PASS / WARNING / FAIL summary and MLflow-ready metrics/artifacts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"

    def __str__(self) -> str:
        return self.value


_STATUS_RANK = {CheckStatus.PASS: 0, CheckStatus.WARNING: 1, CheckStatus.FAIL: 2}


def _worst(*statuses: CheckStatus) -> CheckStatus:
    return max(statuses, key=lambda s: _STATUS_RANK[s]) if statuses else CheckStatus.PASS


@dataclass
class QualityCheckResult:
    name: str
    status: CheckStatus
    message: str
    metric_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": str(self.status),
            "message": self.message,
            "metric_value": self.metric_value,
            "details": self.details,
        }


@dataclass
class ForecastQualityThresholds:
    """Configurable thresholds for WARNING / FAIL levels."""

    # Absolute bias |mean(pred - actual)| — FAIL if above fail, else WARNING if above warn
    bias_warn: float = 50.0
    bias_fail: float = 200.0

    # Relative bias |bias| / mean(|actual|) — optional scale-free gate
    relative_bias_warn: float = 0.10
    relative_bias_fail: float = 0.25

    # Error metrics
    rmse_warn: float = 100.0
    rmse_fail: float = 500.0
    mae_warn: float = 80.0
    mae_fail: float = 400.0
    wape_warn: float = 15.0   # percent
    wape_fail: float = 40.0

    # Abnormal error vs history: error / mean(|actual|)
    relative_rmse_warn: float = 0.25
    relative_rmse_fail: float = 0.50

    # Negative volume / prediction counts
    negative_volume_warn: int = 1
    negative_volume_fail: int = 10
    negative_prediction_warn: int = 1
    negative_prediction_fail: int = 10

    # Missing prediction fraction
    missing_prediction_warn: float = 0.01
    missing_prediction_fail: float = 0.05

    # Timestamp gaps / duplicates
    missing_timestamp_warn: int = 1
    missing_timestamp_fail: int = 5
    duplicate_timestamp_warn: int = 1
    duplicate_timestamp_fail: int = 5

    # Expected timestamp frequency for gap detection (pandas offset alias) or None to infer
    expected_freq: Optional[str] = "D"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ForecastQualityThresholds":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in valid})

    @classmethod
    def from_yaml(
        cls,
        config_path: Union[str, Path] = "/usr/local/airflow/include/config/ml_config.yaml",
    ) -> "ForecastQualityThresholds":
        path = Path(config_path)
        if not path.exists():
            return cls()
        with path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        section = (
            (cfg.get("monitoring") or {}).get("forecast_quality")
            or cfg.get("forecast_quality")
            or {}
        )
        return cls.from_mapping(section)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _threshold_status(
    value: float,
    warn: float,
    fail: float,
    *,
    higher_is_worse: bool = True,
) -> CheckStatus:
    if higher_is_worse:
        if value >= fail:
            return CheckStatus.FAIL
        if value >= warn:
            return CheckStatus.WARNING
        return CheckStatus.PASS
    # lower is worse (unused currently)
    if value <= fail:
        return CheckStatus.FAIL
    if value <= warn:
        return CheckStatus.WARNING
    return CheckStatus.PASS


def _count_status(count: int, warn: int, fail: int) -> CheckStatus:
    if count >= fail:
        return CheckStatus.FAIL
    if count >= warn:
        return CheckStatus.WARNING
    return CheckStatus.PASS


class ForecastQualityChecker:
    """Run automated quality checks on forecast actuals + predictions."""

    def __init__(self, thresholds: Optional[ForecastQualityThresholds] = None):
        self.thresholds = thresholds or ForecastQualityThresholds()

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path] = "/usr/local/airflow/include/config/ml_config.yaml",
    ) -> "ForecastQualityChecker":
        return cls(ForecastQualityThresholds.from_yaml(config_path))

    def check_frame(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "date",
        actual_col: str = "sales",
        prediction_col: str = "prediction",
        model_name: str = "forecast",
    ) -> Dict[str, Any]:
        """
        Run all checks on a DataFrame with timestamps, actuals, and predictions.
        """
        checks: List[QualityCheckResult] = []
        frame = df.copy()
        if timestamp_col in frame.columns:
            frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])

        checks.append(self._check_missing_timestamps(frame, timestamp_col))
        checks.append(self._check_duplicate_timestamps(frame, timestamp_col))
        if actual_col in frame.columns:
            checks.append(
                self._check_negative_values(
                    frame[actual_col],
                    name="negative_volumes",
                    warn=self.thresholds.negative_volume_warn,
                    fail=self.thresholds.negative_volume_fail,
                    label="actual volumes",
                )
            )
        if prediction_col in frame.columns:
            checks.append(
                self._check_negative_values(
                    frame[prediction_col],
                    name="negative_predictions",
                    warn=self.thresholds.negative_prediction_warn,
                    fail=self.thresholds.negative_prediction_fail,
                    label="predictions",
                )
            )
            checks.append(
                self._check_missing_predictions_series(frame[prediction_col], len(frame))
            )
        else:
            checks.append(
                QualityCheckResult(
                    name="missing_predictions",
                    status=CheckStatus.FAIL,
                    message=f"Prediction column '{prediction_col}' is missing",
                    metric_value=1.0,
                )
            )

        if (
            actual_col in frame.columns
            and prediction_col in frame.columns
            and len(frame) > 0
        ):
            y_true = frame[actual_col].to_numpy(dtype=float)
            y_pred = frame[prediction_col].to_numpy(dtype=float)
            checks.extend(self._check_bias_and_errors(y_true, y_pred))

        return self._build_report(model_name, checks)

    def check_models(
        self,
        actual_df: pd.DataFrame,
        predictions: Mapping[str, np.ndarray],
        timestamp_col: str = "date",
        actual_col: str = "sales",
        entity_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run checks for multiple models; return per-model reports + overall summary.

        If ``entity_col`` is set (e.g. store_id), duplicate timestamps are evaluated
        on (timestamp, entity) pairs so panel/multi-series frames are supported.
        Gap detection still uses the unique timestamp calendar.
        """
        model_reports = {}
        ts = actual_df[timestamp_col] if timestamp_col in actual_df.columns else None
        entities = (
            actual_df[entity_col]
            if entity_col and entity_col in actual_df.columns
            else None
        )
        y_true = actual_df[actual_col].to_numpy(dtype=float)

        for model_name, preds in predictions.items():
            if preds is None:
                model_reports[model_name] = self._build_report(
                    model_name,
                    [
                        QualityCheckResult(
                            name="missing_predictions",
                            status=CheckStatus.FAIL,
                            message="Predictions are None",
                            metric_value=1.0,
                        )
                    ],
                )
                continue
            model_reports[model_name] = self.check_arrays(
                y_true=y_true,
                y_pred=np.asarray(preds),
                timestamps=ts,
                entities=entities,
                model_name=model_name,
            )

        overall = _worst(
            *(CheckStatus(r["summary"]) for r in model_reports.values())
        ) if model_reports else CheckStatus.PASS

        return {
            "summary": str(overall),
            "n_models": len(model_reports),
            "models": model_reports,
            "thresholds": self.thresholds.to_dict(),
        }

    def check_arrays(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[Sequence] = None,
        entities: Optional[Sequence] = None,
        model_name: str = "forecast",
    ) -> Dict[str, Any]:
        """Run checks from numpy arrays (optional timestamps / entity keys)."""
        checks: List[QualityCheckResult] = []
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        if timestamps is not None:
            ts = pd.to_datetime(pd.Series(list(timestamps)))
            frame = pd.DataFrame({"timestamp": ts})
            checks.append(self._check_missing_timestamps(frame, "timestamp"))
            if entities is not None:
                frame["entity"] = list(entities)
                checks.append(
                    self._check_duplicate_timestamps(
                        frame, "timestamp", entity_col="entity"
                    )
                )
            else:
                checks.append(self._check_duplicate_timestamps(frame, "timestamp"))
        else:
            checks.append(
                QualityCheckResult(
                    name="missing_timestamps",
                    status=CheckStatus.PASS,
                    message="Timestamp check skipped (no timestamps provided)",
                    metric_value=0.0,
                )
            )
            checks.append(
                QualityCheckResult(
                    name="duplicate_timestamps",
                    status=CheckStatus.PASS,
                    message="Duplicate timestamp check skipped (no timestamps provided)",
                    metric_value=0.0,
                )
            )

        checks.append(
            self._check_negative_values(
                pd.Series(y_true),
                name="negative_volumes",
                warn=self.thresholds.negative_volume_warn,
                fail=self.thresholds.negative_volume_fail,
                label="actual volumes",
            )
        )
        checks.append(
            self._check_negative_values(
                pd.Series(y_pred),
                name="negative_predictions",
                warn=self.thresholds.negative_prediction_warn,
                fail=self.thresholds.negative_prediction_fail,
                label="predictions",
            )
        )
        checks.append(self._check_missing_predictions_arrays(y_true, y_pred))
        checks.extend(self._check_bias_and_errors(y_true, y_pred))

        return self._build_report(model_name, checks)

    # --- individual checks -------------------------------------------------

    def _check_missing_timestamps(
        self, df: pd.DataFrame, timestamp_col: str
    ) -> QualityCheckResult:
        if timestamp_col not in df.columns or df.empty:
            return QualityCheckResult(
                name="missing_timestamps",
                status=CheckStatus.FAIL,
                message=f"Timestamp column '{timestamp_col}' missing or empty",
                metric_value=float("nan"),
            )

        ts = pd.to_datetime(df[timestamp_col]).sort_values()
        nulls = int(ts.isna().sum())
        # Gap detection on the unique calendar (panel frames share dates across entities)
        ts_unique = ts.dropna().drop_duplicates().sort_values()
        if len(ts_unique) < 2:
            return QualityCheckResult(
                name="missing_timestamps",
                status=CheckStatus.WARNING if nulls else CheckStatus.PASS,
                message="Not enough timestamps to detect gaps",
                metric_value=float(nulls),
                details={"null_timestamps": nulls},
            )

        freq = self.thresholds.expected_freq
        if not freq:
            inferred = pd.infer_freq(ts_unique)
            freq = inferred or "D"

        try:
            expected = pd.date_range(ts_unique.min(), ts_unique.max(), freq=freq)
            missing = expected.difference(ts_unique)
            n_missing = int(len(missing)) + nulls
        except Exception as e:
            return QualityCheckResult(
                name="missing_timestamps",
                status=CheckStatus.WARNING,
                message=f"Could not evaluate timestamp gaps ({e})",
                metric_value=float(nulls),
                details={"null_timestamps": nulls, "freq": freq},
            )

        status = _count_status(
            n_missing,
            self.thresholds.missing_timestamp_warn,
            self.thresholds.missing_timestamp_fail,
        )
        return QualityCheckResult(
            name="missing_timestamps",
            status=status,
            message=f"Found {n_missing} missing timestamps (freq={freq})",
            metric_value=float(n_missing),
            details={
                "freq": freq,
                "null_timestamps": nulls,
                "gap_count": int(len(missing)),
                "sample_gaps": [str(x) for x in list(missing[:5])],
            },
        )

    def _check_duplicate_timestamps(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        entity_col: Optional[str] = None,
    ) -> QualityCheckResult:
        if timestamp_col not in df.columns or df.empty:
            return QualityCheckResult(
                name="duplicate_timestamps",
                status=CheckStatus.FAIL,
                message=f"Timestamp column '{timestamp_col}' missing or empty",
                metric_value=float("nan"),
            )
        ts = pd.to_datetime(df[timestamp_col])
        if entity_col and entity_col in df.columns:
            n_dupes = int(df.duplicated(subset=[timestamp_col, entity_col]).sum())
            scope = f"({timestamp_col}, {entity_col})"
        else:
            n_dupes = int(ts.duplicated().sum())
            scope = timestamp_col
        status = _count_status(
            n_dupes,
            self.thresholds.duplicate_timestamp_warn,
            self.thresholds.duplicate_timestamp_fail,
        )
        return QualityCheckResult(
            name="duplicate_timestamps",
            status=status,
            message=f"Found {n_dupes} duplicate timestamps on {scope}",
            metric_value=float(n_dupes),
        )

    def _check_negative_values(
        self,
        series: pd.Series,
        name: str,
        warn: int,
        fail: int,
        label: str,
    ) -> QualityCheckResult:
        vals = pd.to_numeric(series, errors="coerce")
        n_neg = int((vals < 0).sum())
        status = _count_status(n_neg, warn, fail)
        return QualityCheckResult(
            name=name,
            status=status,
            message=f"Found {n_neg} negative {label}",
            metric_value=float(n_neg),
        )

    def _check_missing_predictions_series(
        self, preds: pd.Series, expected_len: int
    ) -> QualityCheckResult:
        vals = pd.to_numeric(preds, errors="coerce").to_numpy(dtype=float)
        n_missing = int((~np.isfinite(vals)).sum())
        frac = n_missing / max(expected_len, 1)
        status = _threshold_status(
            frac,
            self.thresholds.missing_prediction_warn,
            self.thresholds.missing_prediction_fail,
        )
        return QualityCheckResult(
            name="missing_predictions",
            status=status,
            message=f"Missing/non-finite predictions: {n_missing} ({frac:.2%})",
            metric_value=float(frac),
            details={"n_missing": n_missing, "n_total": expected_len},
        )

    def _check_missing_predictions_arrays(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> QualityCheckResult:
        if len(y_pred) != len(y_true):
            return QualityCheckResult(
                name="missing_predictions",
                status=CheckStatus.FAIL,
                message=(
                    f"Prediction length {len(y_pred)} does not match "
                    f"actual length {len(y_true)}"
                ),
                metric_value=1.0,
                details={"n_actual": len(y_true), "n_pred": len(y_pred)},
            )
        n_missing = int((~np.isfinite(y_pred)).sum())
        frac = n_missing / max(len(y_pred), 1)
        status = _threshold_status(
            frac,
            self.thresholds.missing_prediction_warn,
            self.thresholds.missing_prediction_fail,
        )
        return QualityCheckResult(
            name="missing_predictions",
            status=status,
            message=f"Missing/non-finite predictions: {n_missing} ({frac:.2%})",
            metric_value=float(frac),
            details={"n_missing": n_missing, "n_total": len(y_pred)},
        )

    def _check_bias_and_errors(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> List[QualityCheckResult]:
        # Align finite pairs only
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() == 0:
            return [
                QualityCheckResult(
                    name="excessive_bias",
                    status=CheckStatus.FAIL,
                    message="No finite actual/prediction pairs for bias/error checks",
                    metric_value=float("nan"),
                ),
                QualityCheckResult(
                    name="abnormal_errors",
                    status=CheckStatus.FAIL,
                    message="No finite actual/prediction pairs for error checks",
                    metric_value=float("nan"),
                ),
            ]

        yt, yp = y_true[mask], y_pred[mask]
        bias = float(np.mean(yp - yt))
        abs_bias = abs(bias)
        mean_abs_actual = float(np.mean(np.abs(yt))) or 1.0
        rel_bias = abs_bias / mean_abs_actual

        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        mae = float(np.mean(np.abs(yp - yt)))
        wape = float(np.sum(np.abs(yp - yt)) / np.sum(np.abs(yt)) * 100) if np.sum(np.abs(yt)) > 0 else float("nan")
        rel_rmse = rmse / mean_abs_actual

        # Bias: worst of absolute and relative gates
        bias_status = _worst(
            _threshold_status(abs_bias, self.thresholds.bias_warn, self.thresholds.bias_fail),
            _threshold_status(
                rel_bias,
                self.thresholds.relative_bias_warn,
                self.thresholds.relative_bias_fail,
            ),
        )
        bias_check = QualityCheckResult(
            name="excessive_bias",
            status=bias_status,
            message=(
                f"Forecast bias={bias:.4f} (|bias|={abs_bias:.4f}, "
                f"relative={rel_bias:.2%})"
            ),
            metric_value=bias,
            details={
                "abs_bias": abs_bias,
                "relative_bias": rel_bias,
                "n_points": int(mask.sum()),
            },
        )

        error_status = _worst(
            _threshold_status(rmse, self.thresholds.rmse_warn, self.thresholds.rmse_fail),
            _threshold_status(mae, self.thresholds.mae_warn, self.thresholds.mae_fail),
            _threshold_status(
                wape if np.isfinite(wape) else 0.0,
                self.thresholds.wape_warn,
                self.thresholds.wape_fail,
            ),
            _threshold_status(
                rel_rmse,
                self.thresholds.relative_rmse_warn,
                self.thresholds.relative_rmse_fail,
            ),
        )
        error_check = QualityCheckResult(
            name="abnormal_errors",
            status=error_status,
            message=(
                f"Errors RMSE={rmse:.4f}, MAE={mae:.4f}, WAPE={wape:.2f}%, "
                f"rel_RMSE={rel_rmse:.2%}"
            ),
            metric_value=rmse,
            details={
                "rmse": rmse,
                "mae": mae,
                "wape": wape,
                "relative_rmse": rel_rmse,
                "n_points": int(mask.sum()),
            },
        )
        return [bias_check, error_check]

    def _build_report(
        self, model_name: str, checks: List[QualityCheckResult]
    ) -> Dict[str, Any]:
        summary = _worst(*(c.status for c in checks)) if checks else CheckStatus.PASS
        report = {
            "model": model_name,
            "summary": str(summary),
            "n_checks": len(checks),
            "n_pass": sum(1 for c in checks if c.status == CheckStatus.PASS),
            "n_warning": sum(1 for c in checks if c.status == CheckStatus.WARNING),
            "n_fail": sum(1 for c in checks if c.status == CheckStatus.FAIL),
            "checks": [c.to_dict() for c in checks],
        }
        logger.info(
            f"Forecast quality [{model_name}]: {summary} "
            f"(pass={report['n_pass']}, warn={report['n_warning']}, fail={report['n_fail']})"
        )
        return report


def flatten_quality_metrics(report: Dict[str, Any], prefix: str = "fq") -> Dict[str, float]:
    """
    Flatten a single-model or multi-model quality report into MLflow metrics.

    Status encoded as: PASS=0, WARNING=1, FAIL=2.
    """
    status_code = {
        "PASS": 0.0,
        "WARNING": 1.0,
        "FAIL": 2.0,
    }
    metrics: Dict[str, float] = {}

    if "models" in report:
        metrics[f"{prefix}_overall_status"] = status_code.get(report.get("summary", "PASS"), 2.0)
        for model_name, model_report in report["models"].items():
            metrics.update(
                flatten_quality_metrics(model_report, prefix=f"{prefix}_{model_name}")
            )
        return metrics

    model = report.get("model", "forecast")
    metrics[f"{prefix}_status"] = status_code.get(report.get("summary", "PASS"), 2.0)
    metrics[f"{prefix}_n_pass"] = float(report.get("n_pass", 0))
    metrics[f"{prefix}_n_warning"] = float(report.get("n_warning", 0))
    metrics[f"{prefix}_n_fail"] = float(report.get("n_fail", 0))
    for check in report.get("checks", []):
        name = check["name"]
        metrics[f"{prefix}_{name}_status"] = status_code.get(check["status"], 2.0)
        if check.get("metric_value") is not None and np.isfinite(check["metric_value"]):
            metrics[f"{prefix}_{name}_value"] = float(check["metric_value"])
    # silence unused
    _ = model
    return metrics


def write_quality_artifacts(
    report: Dict[str, Any],
    output_dir: str = "/tmp/forecast_quality",
) -> Dict[str, str]:
    """Persist JSON + CSV summary artifacts; return file paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    json_path = os.path.join(output_dir, "forecast_quality_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    paths["json"] = json_path

    # Flat check table
    rows = []
    if "models" in report:
        for model_name, model_report in report["models"].items():
            for check in model_report.get("checks", []):
                rows.append(
                    {
                        "model": model_name,
                        "model_summary": model_report.get("summary"),
                        "check": check["name"],
                        "status": check["status"],
                        "message": check["message"],
                        "metric_value": check.get("metric_value"),
                    }
                )
            rows.append(
                {
                    "model": model_name,
                    "model_summary": model_report.get("summary"),
                    "check": "SUMMARY",
                    "status": model_report.get("summary"),
                    "message": (
                        f"pass={model_report.get('n_pass')}, "
                        f"warn={model_report.get('n_warning')}, "
                        f"fail={model_report.get('n_fail')}"
                    ),
                    "metric_value": None,
                }
            )
        rows.append(
            {
                "model": "ALL",
                "model_summary": report.get("summary"),
                "check": "OVERALL",
                "status": report.get("summary"),
                "message": f"n_models={report.get('n_models')}",
                "metric_value": None,
            }
        )
    else:
        for check in report.get("checks", []):
            rows.append(
                {
                    "model": report.get("model"),
                    "model_summary": report.get("summary"),
                    "check": check["name"],
                    "status": check["status"],
                    "message": check["message"],
                    "metric_value": check.get("metric_value"),
                }
            )

    csv_path = os.path.join(output_dir, "forecast_quality_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    # Human-readable summary text
    txt_path = os.path.join(output_dir, "forecast_quality_summary.txt")
    lines = [f"FORECAST QUALITY SUMMARY: {report.get('summary', 'PASS')}", ""]
    if "models" in report:
        for model_name, model_report in report["models"].items():
            lines.append(f"[{model_report.get('summary')}] {model_name}")
            for check in model_report.get("checks", []):
                lines.append(f"  - {check['status']}: {check['name']} — {check['message']}")
            lines.append("")
    else:
        lines.append(f"[{report.get('summary')}] {report.get('model')}")
        for check in report.get("checks", []):
            lines.append(f"  - {check['status']}: {check['name']} — {check['message']}")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    paths["txt"] = txt_path

    return paths
