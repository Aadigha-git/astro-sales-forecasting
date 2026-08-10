import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import yaml
import joblib
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import xgboost as xgb
import lightgbm as lgb
import optuna
import mlflow

from utils.mlflow_utils import MLflowManager
from feature_engineering.feature_pipeline import FeatureEngineer
from data_validation.validators import DataValidator
from ml_models.advanced_ensemble import AdvancedEnsemble
from ml_models.diagnostics import diagnose_model_performance
from ml_models.ensemble_model import EnsembleModel
from ml_models.classical_ts_models import (
    SeasonalNaiveModel,
    HoltWintersModel,
    SARIMAXModel,
    build_model_from_config,
)
from ml_models.rolling_origin_cv import RollingOriginEvaluator
from ml_models.horizon_evaluation import (
    MultiHorizonEvaluator,
    DEFAULT_HORIZONS,
    horizon_summary_frame,
)
from ml_models.forecast_quality import (
    ForecastQualityChecker,
    flatten_quality_metrics,
    write_quality_artifacts,
)

logger = logging.getLogger(__name__)

# Models that participate in comparison / ensemble evaluation when enabled
CLASSICAL_TS_MODELS = ("seasonal_naive", "holt_winters", "sarimax")


class ModelTrainer:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']
        self.training_config = self.config['training']
        self.mlflow_manager = MLflowManager(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.data_validator = DataValidator(config_path)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'sales',
                    date_col: str = 'date', group_cols: Optional[List[str]] = None,
                    categorical_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        logger.info("Preparing data for training")
        
        # Skip validation for aggregated data that doesn't include product_id
        # The validation expects product-level data but we're training on store-level aggregates
        # Validate only essential columns
        required_cols = ['date', target_col]
        if group_cols:
            required_cols.extend(group_cols)
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {missing_cols}")
        
        # Feature engineering
        df_features = self.feature_engineer.create_all_features(
            df, target_col, date_col, group_cols, categorical_cols
        )
        
        # Skip target encoding to avoid data leakage
        # Target encoding can cause overfitting in time series
        
        # Split data chronologically for time series
        df_sorted = df_features.sort_values(date_col)
        
        # Use more recent data for validation and testing for better performance
        # This ensures the model learns from patterns closer to what it will predict
        train_size = int(len(df_sorted) * (1 - self.training_config['test_size'] - self.training_config['validation_size']))
        val_size = int(len(df_sorted) * self.training_config['validation_size'])
        
        train_df = df_sorted[:train_size]
        val_df = df_sorted[train_size:train_size + val_size]
        test_df = df_sorted[train_size + val_size:]
        
        # Remove any rows with NaN in target column
        train_df = train_df.dropna(subset=[target_col])
        val_df = val_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])
        
        # Skip feature selection for now - let models handle all features
        # This allows models to learn which features are important
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def preprocess_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, target_col: str,
                           exclude_cols: List[str] = ['date']) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col not in exclude_cols + [target_col]]
        
        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()
        
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        
        # Encode categorical variables
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_train.loc[:, col] = self.encoders[col].fit_transform(X_train[col].astype(str))
            else:
                X_train.loc[:, col] = self.encoders[col].transform(X_train[col].astype(str))
            
            X_val.loc[:, col] = self.encoders[col].transform(X_val[col].astype(str))
            X_test.loc[:, col] = self.encoders[col].transform(X_test[col].astype(str))
        
        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        self.scalers['standard'] = scaler
        self.feature_cols = feature_cols
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        abs_err = np.abs(y_true - y_pred)
        actual_sum = np.sum(np.abs(y_true))
        # WAPE: sum(|error|) / sum(|actual|) * 100
        wape = float(np.sum(abs_err) / actual_sum * 100) if actual_sum > 0 else float("nan")
        # Forecast bias: mean(pred - actual); positive => over-forecast
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
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     use_optuna: bool = True) -> xgb.XGBRegressor:
        
        logger.info("Training XGBoost model")
        
        if use_optuna:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42
                }
                
                params['early_stopping_rounds'] = 50
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                y_pred = model.predict(X_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials', 50))
            
            best_params = study.best_params
            best_params['random_state'] = 42
        else:
            best_params = self.model_config['xgboost']['params']
        
        best_params['early_stopping_rounds'] = 50
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      use_optuna: bool = True) -> lgb.LGBMRegressor:
        
        logger.info("Training LightGBM model")
        
        if use_optuna:
            def objective(trial):
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42,
                    'verbosity': -1,
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt'
                }
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                
                y_pred = model.predict(X_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials', 50))
            
            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['verbosity'] = -1
        else:
            best_params = self.model_config['lightgbm']['params']
        
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                 callbacks=[lgb.early_stopping(50)])
        
        self.models['lightgbm'] = model
        return model
    
    def train_prophet(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                     date_col: str = 'date', target_col: str = 'sales') -> Any:
        
        from prophet import Prophet

        logger.info("Training Prophet model")
        
        # Prepare data for Prophet
        prophet_train = train_df[[date_col, target_col]].rename(
            columns={date_col: 'ds', target_col: 'y'}
        )
        
        # Remove any NaN values
        prophet_train = prophet_train.dropna()
        
        # Ensure dates are sorted
        prophet_train = prophet_train.sort_values('ds')
        
        # Initialize Prophet with simplified parameters to avoid memory issues
        prophet_params = self.model_config['prophet']['params'].copy()
        
        # Override some parameters for stability
        prophet_params.update({
            'stan_backend': 'CMDSTANPY',  # Use cmdstanpy backend
            'mcmc_samples': 0,  # Disable MCMC for speed and stability
            'uncertainty_samples': 100,  # Reduce uncertainty samples
        })
        
        try:
            model = Prophet(**prophet_params)
            
            # Add only essential regressors to reduce complexity
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            regressor_cols = [col for col in numeric_cols if col not in [target_col, 'year', 'month', 'day', 'week', 'quarter']]
            
            # Limit to top 5 most important regressors based on variance
            if len(regressor_cols) > 5:
                variances = {col: train_df[col].var() for col in regressor_cols}
                regressor_cols = sorted(variances.keys(), key=lambda x: variances[x], reverse=True)[:5]
            
            for col in regressor_cols:
                if train_df[col].std() > 0:  # Only add regressors with variance
                    model.add_regressor(col)
                    prophet_train[col] = train_df[col]
            
            # Fit the model with error handling
            model.fit(prophet_train)
            
            self.models['prophet'] = model
            return model
            
        except Exception as e:
            logger.error(f"Prophet training failed with parameters: {e}")
            # Try with minimal configuration
            logger.info("Retrying Prophet with minimal configuration...")
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                uncertainty_samples=50,
                mcmc_samples=0
            )
            
            # Train without any additional regressors
            model.fit(prophet_train[['ds', 'y']])
            
            self.models['prophet'] = model
            return model

    def train_seasonal_naive(
        self, y_history: np.ndarray, season_length: Optional[int] = None
    ) -> SeasonalNaiveModel:
        logger.info("Training Seasonal Naive model")
        params = dict(self.model_config.get("seasonal_naive", {}).get("params", {}))
        if season_length is not None:
            params["season_length"] = season_length
        model = build_model_from_config("seasonal_naive", params)
        model.fit(y_history)
        self.models["seasonal_naive"] = model
        return model

    def train_holt_winters(self, y_history: np.ndarray) -> HoltWintersModel:
        logger.info("Training Holt-Winters (Exponential Smoothing) model")
        params = dict(self.model_config.get("holt_winters", {}).get("params", {}))
        model = build_model_from_config("holt_winters", params)
        model.fit(y_history)
        self.models["holt_winters"] = model
        return model

    def train_sarimax(self, y_history: np.ndarray) -> SARIMAXModel:
        logger.info("Training SARIMAX model")
        params = dict(self.model_config.get("sarimax", {}).get("params", {}))
        model = build_model_from_config("sarimax", params)
        model.fit(y_history)
        self.models["sarimax"] = model
        return model

    def _is_model_enabled(self, model_name: str, default: bool = False) -> bool:
        return bool(self.model_config.get(model_name, {}).get("enabled", default))

    def _train_classical_ts_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_test: np.ndarray,
        target_col: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Train enabled classical TS baselines and return results for comparison/ensemble."""
        classical_results: Dict[str, Dict[str, Any]] = {}
        y_train = train_df[target_col].astype(float).values
        y_train_val = pd.concat([train_df, val_df], ignore_index=True)[target_col].astype(float).values
        n_val = len(val_df)
        n_test = len(test_df)

        trainers = {
            "seasonal_naive": self.train_seasonal_naive,
            "holt_winters": self.train_holt_winters,
            "sarimax": self.train_sarimax,
        }

        for model_name, train_fn in trainers.items():
            if not self._is_model_enabled(model_name, default=False):
                logger.info(f"{model_name} disabled in config; skipping")
                continue

            try:
                # Val forecast from train-only fit (for ensemble weights)
                model_val = train_fn(y_train)
                val_pred = model_val.predict(n_val)

                # Test forecast from train+val fit (more history for final evaluation)
                model = train_fn(y_train_val)
                test_pred = model.predict(n_test)
                metrics = self.calculate_metrics(y_test, test_pred)

                self.mlflow_manager.log_metrics(
                    {f"{model_name}_{k}": v for k, v in metrics.items()}
                )
                try:
                    self.mlflow_manager.log_model(model, model_name)
                except Exception as log_err:
                    logger.warning(f"Could not log {model_name} to MLflow as model artifact: {log_err}")

                classical_results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "predictions": test_pred,
                    "val_predictions": val_pred,
                }
                logger.info(
                    f"{model_name} metrics - RMSE: {metrics['rmse']:.4f}, "
                    f"MAE: {metrics['mae']:.4f}, WAPE: {metrics['wape']:.4f}, "
                    f"Bias: {metrics['bias']:.4f}, R2: {metrics['r2']:.4f}"
                )
            except Exception as e:
                logger.warning(f"{model_name} training failed: {e}", exc_info=True)

        return classical_results

    @staticmethod
    def _compute_ensemble_weights(
        val_scores: Dict[str, float], min_weight: float = 0.05
    ) -> Dict[str, float]:
        """Convert validation R² scores into normalized ensemble weights."""
        # Shift R² into a positive range so poor models still get a small weight
        shifted = {name: max(0.0, score) + 1e-6 for name, score in val_scores.items()}
        total = sum(shifted.values())
        weights = {name: max(min_weight, score / total) for name, score in shifted.items()}
        weight_sum = sum(weights.values())
        return {name: w / weight_sum for name, w in weights.items()}

    def run_rolling_origin_cv(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
    ) -> Optional[Dict[str, Any]]:
        """
        Rolling-origin CV on train+validation only (final test holdout preserved).

        Returns aggregated + per-fold metrics for each evaluated model, or None if disabled.
        """
        cv_cfg = dict(self.training_config.get("rolling_origin_cv", {}) or {})
        if not cv_cfg.get("enabled", False):
            logger.info("Rolling-origin CV disabled in config")
            return None

        cv_data = pd.concat([train_df, val_df], ignore_index=True)
        cv_data = cv_data.sort_values(date_col).reset_index(drop=True)

        n_splits = int(cv_cfg.get("n_splits") or self.training_config.get("cv_folds", 5))
        horizon = cv_cfg.get("horizon")
        if horizon is None:
            horizon = self.config.get("inference", {}).get("prediction_horizon", 30)
        horizon = int(horizon)
        step = cv_cfg.get("step")
        step = int(step) if step is not None else None
        min_train_size = cv_cfg.get("min_train_size")
        min_train_size = int(min_train_size) if min_train_size is not None else None

        self.mlflow_manager.log_params(
            {
                "cv_n_splits": n_splits,
                "cv_horizon": horizon,
                "cv_step": step if step is not None else horizon,
                "cv_n_samples": len(cv_data),
            }
        )

        evaluator = RollingOriginEvaluator(
            model_config=self.model_config,
            metrics_fn=self.calculate_metrics,
        )

        try:
            cv_results = evaluator.evaluate(
                df=cv_data,
                target_col=target_col,
                date_col=date_col,
                n_splits=n_splits,
                horizon=horizon,
                min_train_size=min_train_size,
                step=step,
            )
        except Exception as e:
            logger.warning(f"Rolling-origin CV failed: {e}", exc_info=True)
            return None

        # Log aggregated metrics to MLflow
        flat_metrics = {}
        for model_name, model_cv in cv_results.get("models", {}).items():
            for metric_name, value in model_cv.get("aggregated", {}).items():
                flat_metrics[f"{model_name}_cv_{metric_name}"] = float(value)
        if flat_metrics:
            self.mlflow_manager.log_metrics(flat_metrics)

        # Persist fold table for later inspection
        try:
            import os

            rows = []
            for model_name, model_cv in cv_results.get("models", {}).items():
                for fold in model_cv.get("folds", []):
                    row = {
                        "model": model_name,
                        "fold_id": fold["fold_id"],
                        "train_end": fold["train_end"],
                        "test_start": fold["test_start"],
                        "test_end": fold["test_end"],
                        "horizon": fold["horizon"],
                        "train_size": fold["train_size"],
                    }
                    row.update(fold.get("metrics", {}))
                    rows.append(row)
            if rows:
                fold_df = pd.DataFrame(rows)
                cv_dir = "/tmp/rolling_origin_cv"
                os.makedirs(cv_dir, exist_ok=True)
                fold_path = os.path.join(cv_dir, "fold_metrics.csv")
                fold_df.to_csv(fold_path, index=False)
                agg_rows = []
                for model_name, model_cv in cv_results.get("models", {}).items():
                    agg = {"model": model_name}
                    agg.update(model_cv.get("aggregated", {}))
                    agg_rows.append(agg)
                agg_path = os.path.join(cv_dir, "aggregated_metrics.csv")
                pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
                self.mlflow_manager.log_artifacts(cv_dir)
        except Exception as art_err:
            logger.warning(f"Could not persist rolling-origin CV artifacts: {art_err}")

        return cv_results

    def run_multi_horizon_evaluation(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate accuracy and bias at 1 / 3 / 7 / 14-day horizons on train+val.

        Holdout test remains untouched. Results include per-model horizon summaries
        suitable for MLflow logging and horizon-vs-accuracy charts.
        """
        mh_cfg = dict(self.training_config.get("multi_horizon_evaluation", {}) or {})
        if not mh_cfg.get("enabled", True):
            logger.info("Multi-horizon evaluation disabled in config")
            return None

        horizons = mh_cfg.get("horizons") or list(DEFAULT_HORIZONS)
        horizons = [int(h) for h in horizons]
        n_splits = int(
            mh_cfg.get("n_splits")
            or self.training_config.get("rolling_origin_cv", {}).get("n_splits")
            or self.training_config.get("cv_folds", 5)
        )
        step = mh_cfg.get("step")
        step = int(step) if step is not None else None
        min_train_size = mh_cfg.get("min_train_size")
        min_train_size = int(min_train_size) if min_train_size is not None else None
        aggregate_daily = bool(mh_cfg.get("aggregate_daily", True))

        cv_data = pd.concat([train_df, val_df], ignore_index=True)

        self.mlflow_manager.log_params(
            {
                "mh_horizons": ",".join(str(h) for h in horizons),
                "mh_n_splits": n_splits,
                "mh_aggregate_daily": aggregate_daily,
            }
        )

        evaluator = MultiHorizonEvaluator(
            model_config=self.model_config,
            metrics_fn=self.calculate_metrics,
        )
        try:
            horizon_results = evaluator.evaluate_horizons(
                df=cv_data,
                target_col=target_col,
                date_col=date_col,
                horizons=horizons,
                n_splits=n_splits,
                step=step,
                min_train_size=min_train_size,
                aggregate_daily=aggregate_daily,
            )
        except Exception as e:
            logger.warning(f"Multi-horizon evaluation failed: {e}", exc_info=True)
            return None

        # Log accuracy + bias by horizon to MLflow
        flat_metrics = {}
        for model_name, payload in horizon_results.get("models", {}).items():
            for row in payload.get("horizon_summary", []):
                h = int(row["horizon"])
                for metric in ("rmse", "mae", "wape", "bias", "mape"):
                    val = row.get(metric)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        flat_metrics[f"{model_name}_h{h}_{metric}"] = float(val)
        if flat_metrics:
            self.mlflow_manager.log_metrics(flat_metrics)

        # Persist summary tables
        try:
            import os

            mh_dir = "/tmp/multi_horizon_evaluation"
            os.makedirs(mh_dir, exist_ok=True)
            summary_df = horizon_summary_frame(horizon_results)
            summary_path = os.path.join(mh_dir, "accuracy_bias_by_horizon.csv")
            summary_df.to_csv(summary_path, index=False)
            # Bias-focused table
            bias_cols = [c for c in ["model", "horizon", "bias", "bias_std", "n_folds"] if c in summary_df.columns]
            if bias_cols:
                summary_df[bias_cols].to_csv(
                    os.path.join(mh_dir, "bias_by_horizon.csv"), index=False
                )
            self.mlflow_manager.log_artifacts(mh_dir)
        except Exception as art_err:
            logger.warning(f"Could not persist multi-horizon artifacts: {art_err}")

        return horizon_results

    def run_forecast_quality_checks(
        self,
        test_df: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
        target_col: str = "sales",
        date_col: str = "date",
    ) -> Optional[Dict[str, Any]]:
        """
        Automated forecast quality checks with PASS / WARNING / FAIL summary.

        Logs metrics + JSON/CSV/TXT artifacts to MLflow. Does not alter training.
        """
        fq_cfg = dict(
            (self.config.get("monitoring") or {}).get("forecast_quality", {}) or {}
        )
        if not fq_cfg.get("enabled", True):
            logger.info("Forecast quality checks disabled in config")
            return None

        try:
            from ml_models.forecast_quality import ForecastQualityThresholds

            checker = ForecastQualityChecker(
                thresholds=ForecastQualityThresholds.from_mapping(fq_cfg)
            )
            pred_map = {
                name: preds
                for name, preds in predictions.items()
                if preds is not None
                and name
                not in (
                    "rolling_origin_cv",
                    "multi_horizon_evaluation",
                    "forecast_quality",
                )
            }
            entity_col = "store_id" if "store_id" in test_df.columns else None
            report = checker.check_models(
                actual_df=test_df,
                predictions=pred_map,
                timestamp_col=date_col,
                actual_col=target_col,
                entity_col=entity_col,
            )

            metrics = flatten_quality_metrics(report)
            if metrics:
                self.mlflow_manager.log_metrics(metrics)

            paths = write_quality_artifacts(report, output_dir="/tmp/forecast_quality")
            self.mlflow_manager.log_artifacts("/tmp/forecast_quality")
            report["artifact_paths"] = paths

            logger.info(
                f"Forecast quality overall: {report.get('summary')} "
                f"(artifacts: {list(paths.values())})"
            )
            return report
        except Exception as e:
            logger.warning(f"Forecast quality checks failed: {e}", exc_info=True)
            return None

    def train_all_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: pd.DataFrame, target_col: str = 'sales',
                        use_optuna: bool = True) -> Dict[str, Dict[str, Any]]:
        
        results = {}
        
        # Start MLflow run
        run_id = self.mlflow_manager.start_run(
            run_name=f"sales_forecast_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={"model_type": "ensemble", "use_optuna": str(use_optuna)}
        )
        
        try:
            # Preprocess data
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_features(
                train_df, val_df, test_df, target_col
            )
            
            # Log data stats
            self.mlflow_manager.log_params({
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "n_features": X_train.shape[1]
            })
            
            # Train XGBoost
            xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val, use_optuna)
            xgb_pred = xgb_model.predict(X_test)
            xgb_metrics = self.calculate_metrics(y_test, xgb_pred)
            
            self.mlflow_manager.log_metrics({f"xgboost_{k}": v for k, v in xgb_metrics.items()})
            self.mlflow_manager.log_model(xgb_model, "xgboost", 
                                         input_example=X_train.iloc[:5])
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            logger.info(f"Top XGBoost features:\n{feature_importance.to_string()}")
            self.mlflow_manager.log_params({f"xgb_top_feature_{i}": f"{row['feature']} ({row['importance']:.4f})" 
                                           for i, (_, row) in enumerate(feature_importance.iterrows())})
            
            results['xgboost'] = {
                'model': xgb_model,
                'metrics': xgb_metrics,
                'predictions': xgb_pred
            }
            
            # Train LightGBM
            lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val, use_optuna)
            lgb_pred = lgb_model.predict(X_test)
            lgb_metrics = self.calculate_metrics(y_test, lgb_pred)
            
            self.mlflow_manager.log_metrics({f"lightgbm_{k}": v for k, v in lgb_metrics.items()})
            self.mlflow_manager.log_model(lgb_model, "lightgbm",
                                         input_example=X_train.iloc[:5])
            
            # Log feature importance for LightGBM
            lgb_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            logger.info(f"Top LightGBM features:\n{lgb_importance.to_string()}")
            
            results['lightgbm'] = {
                'model': lgb_model,
                'metrics': lgb_metrics,
                'predictions': lgb_pred
            }

            # Validation predictions for ensemble weighting
            xgb_val_pred = xgb_model.predict(X_val)
            lgb_val_pred = lgb_model.predict(X_val)

            ensemble_member_preds = {
                'xgboost': {'test': xgb_pred, 'val': xgb_val_pred},
                'lightgbm': {'test': lgb_pred, 'val': lgb_val_pred},
            }

            # Train Prophet if enabled
            prophet_enabled = self._is_model_enabled('prophet', default=False)

            if prophet_enabled:
                try:
                    prophet_model = self.train_prophet(train_df, val_df)

                    # Create future dataframe for Prophet predictions
                    future = test_df[['date']].rename(columns={'date': 'ds'})

                    # Add regressors if they exist
                    if hasattr(prophet_model, 'extra_regressors') and prophet_model.extra_regressors:
                        regressor_cols = [col for col in prophet_model.extra_regressors.keys()]
                        for col in regressor_cols:
                            if col in test_df.columns:
                                future[col] = test_df[col]

                    prophet_pred = prophet_model.predict(future)['yhat'].values
                    prophet_metrics = self.calculate_metrics(y_test, prophet_pred)

                    self.mlflow_manager.log_metrics({f"prophet_{k}": v for k, v in prophet_metrics.items()})

                    # Val predictions for weighting
                    future_val = val_df[['date']].rename(columns={'date': 'ds'})
                    if hasattr(prophet_model, 'extra_regressors') and prophet_model.extra_regressors:
                        for col in prophet_model.extra_regressors.keys():
                            if col in val_df.columns:
                                future_val[col] = val_df[col]
                    prophet_val_pred = prophet_model.predict(future_val)['yhat'].values

                    results['prophet'] = {
                        'model': prophet_model,
                        'metrics': prophet_metrics,
                        'predictions': prophet_pred
                    }
                    ensemble_member_preds['prophet'] = {
                        'test': prophet_pred,
                        'val': prophet_val_pred,
                    }
                except Exception as e:
                    logger.warning(
                        f"Prophet training failed: {e}. Continuing without Prophet."
                    )
                    prophet_enabled = False

            # Classical time-series models (Seasonal Naive, Holt-Winters, SARIMAX)
            classical_results = self._train_classical_ts_models(
                train_df, val_df, test_df, y_test, target_col
            )
            for model_name, model_result in classical_results.items():
                results[model_name] = {
                    'model': model_result['model'],
                    'metrics': model_result['metrics'],
                    'predictions': model_result['predictions'],
                }
                ensemble_member_preds[model_name] = {
                    'test': model_result['predictions'],
                    'val': model_result['val_predictions'],
                }

            # Weighted ensemble from all successfully trained members (evaluation)
            val_scores = {
                name: r2_score(y_val, preds['val'])
                for name, preds in ensemble_member_preds.items()
            }
            ensemble_weights = self._compute_ensemble_weights(val_scores)
            logger.info(
                "Ensemble weights: "
                + ", ".join(f"{k}: {v:.3f}" for k, v in ensemble_weights.items())
            )

            ensemble_pred = np.zeros_like(y_test, dtype=float)
            for name, preds in ensemble_member_preds.items():
                ensemble_pred += ensemble_weights[name] * np.asarray(preds['test'], dtype=float)

            # Deployable EnsembleModel keeps tabular models only (predict(X) API)
            deployable_models = {
                'xgboost': xgb_model,
                'lightgbm': lgb_model,
            }
            deployable_weights = {
                k: ensemble_weights[k]
                for k in deployable_models
                if k in ensemble_weights
            }
            # Renormalize deployable weights
            deployable_total = sum(deployable_weights.values()) or 1.0
            deployable_weights = {
                k: v / deployable_total for k, v in deployable_weights.items()
            }

            ensemble_model = EnsembleModel(deployable_models, deployable_weights)

            # Save ensemble model
            self.models['ensemble'] = ensemble_model

            ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred)

            self.mlflow_manager.log_metrics({f"ensemble_{k}": v for k, v in ensemble_metrics.items()})
            self.mlflow_manager.log_params(
                {f"ensemble_weight_{k}": float(v) for k, v in ensemble_weights.items()}
            )
            self.mlflow_manager.log_model(ensemble_model, "ensemble",
                                         input_example=X_train.iloc[:5])

            results['ensemble'] = {
                'model': ensemble_model,
                'metrics': ensemble_metrics,
                'predictions': ensemble_pred,
                'weights': ensemble_weights,
            }

            # Rolling-origin CV on train+val (holdout test unchanged)
            logger.info("Running rolling-origin time-series cross-validation...")
            cv_results = self.run_rolling_origin_cv(
                train_df, val_df, target_col=target_col, date_col="date"
            )
            if cv_results:
                results["rolling_origin_cv"] = cv_results
                for model_name, model_cv in cv_results.get("models", {}).items():
                    if model_name in results:
                        results[model_name]["cv_metrics"] = model_cv.get("aggregated", {})

            # Multi-horizon evaluation (1 / 3 / 7 / 14 day)
            logger.info("Running multi-horizon forecast evaluation...")
            horizon_results = self.run_multi_horizon_evaluation(
                train_df, val_df, target_col=target_col, date_col="date"
            )
            if horizon_results:
                results["multi_horizon_evaluation"] = horizon_results
                for model_name, payload in horizon_results.get("models", {}).items():
                    if model_name in results:
                        results[model_name]["horizon_metrics"] = payload.get(
                            "horizon_summary", []
                        )

            # Run diagnostics
            logger.info("Running model diagnostics...")
            test_predictions = {
                name: result['predictions']
                for name, result in results.items()
                if isinstance(result, dict) and result.get('predictions') is not None
            }

            diagnosis = diagnose_model_performance(
                train_df, val_df, test_df, test_predictions, target_col
            )

            logger.info("Diagnostic recommendations:")
            for rec in diagnosis['recommendations']:
                logger.warning(f"- {rec}")

            # Automated forecast quality checks (PASS / WARNING / FAIL)
            logger.info("Running automated forecast quality checks...")
            quality_report = self.run_forecast_quality_checks(
                test_df, test_predictions, target_col=target_col, date_col="date"
            )
            if quality_report:
                results["forecast_quality"] = {
                    "summary": quality_report.get("summary"),
                    "n_models": quality_report.get("n_models"),
                    "models": {
                        name: {
                            "summary": rep.get("summary"),
                            "n_pass": rep.get("n_pass"),
                            "n_warning": rep.get("n_warning"),
                            "n_fail": rep.get("n_fail"),
                            "checks": rep.get("checks"),
                        }
                        for name, rep in (quality_report.get("models") or {}).items()
                    },
                }

            # Generate visualizations
            logger.info("Generating model comparison visualizations...")
            try:
                self._generate_and_log_visualizations(
                    results,
                    test_df,
                    target_col,
                    cv_results=cv_results,
                    horizon_results=horizon_results,
                )
            except Exception as viz_error:
                logger.error(f"Visualization generation failed: {viz_error}", exc_info=True)

            # Save artifacts
            self.save_artifacts()

            # Get current run ID for verification
            current_run_id = mlflow.active_run().info.run_id

            self.mlflow_manager.end_run()

            # Sync artifacts to S3
            from utils.mlflow_s3_utils import MLflowS3Manager

            logger.info("Syncing artifacts to S3...")
            try:
                s3_manager = MLflowS3Manager()
                s3_manager.sync_mlflow_artifacts_to_s3(current_run_id)
                logger.info("✓ Successfully synced artifacts to S3")

                # Verify S3 artifacts after sync
                from utils.s3_verification import verify_s3_artifacts, log_s3_verification_results

                logger.info("Verifying S3 artifact storage...")
                verification_results = verify_s3_artifacts(
                    run_id=current_run_id,
                    expected_artifacts=[
                        'models/',
                        'scalers.pkl',
                        'encoders.pkl',
                        'feature_cols.pkl',
                        'visualizations/',
                        'reports/'
                    ]
                )
                log_s3_verification_results(verification_results)

                if not verification_results["success"]:
                    logger.warning("S3 artifact verification failed after sync")
            except Exception as e:
                logger.error(f"Failed to sync artifacts to S3: {e}")

        except Exception as e:
            self.mlflow_manager.end_run(status="FAILED")
            raise e

        return results

    def _generate_and_log_visualizations(self, results: Dict[str, Any],
                                       test_df: pd.DataFrame,
                                       target_col: str = 'sales',
                                       cv_results: Optional[Dict[str, Any]] = None,
                                       horizon_results: Optional[Dict[str, Any]] = None) -> None:
        """Generate and log model comparison visualizations to MLflow"""
        try:
            from ml_models.model_visualization import ModelVisualizer
            import tempfile
            import os
            
            logger.info("Starting visualization generation...")
            visualizer = ModelVisualizer()
            
            # Extract metrics
            metrics_dict = {}
            for model_name, model_results in results.items():
                if not isinstance(model_results, dict):
                    continue
                if model_name == "rolling_origin_cv":
                    continue
                if model_name == "multi_horizon_evaluation":
                    continue
                if model_name == "forecast_quality":
                    continue
                if "metrics" in model_results:
                    metrics_dict[model_name] = model_results["metrics"]

            # Prepare predictions data
            predictions_dict = {}
            for model_name, model_results in results.items():
                if not isinstance(model_results, dict):
                    continue
                if "predictions" in model_results and model_results["predictions"] is not None:
                    pred_df = test_df[["date"]].copy()
                    pred_df["prediction"] = model_results["predictions"]
                    predictions_dict[model_name] = pred_df

            # Extract feature importance if available
            feature_importance_dict = {}
            for model_name, model_results in results.items():
                if model_name in ["xgboost", "lightgbm"] and isinstance(model_results, dict) and "model" in model_results:
                    model = model_results["model"]
                    if hasattr(model, "feature_importances_"):
                        importance_df = pd.DataFrame({
                            "feature": self.feature_cols,
                            "importance": model.feature_importances_,
                        }).sort_values("importance", ascending=False)
                        feature_importance_dict[model_name] = importance_df

            # Create temporary directory for visualizations
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Creating visualizations in temporary directory: {temp_dir}")

                # Generate all visualizations
                saved_files = visualizer.create_comprehensive_report(
                    metrics_dict=metrics_dict,
                    predictions_dict=predictions_dict,
                    actual_data=test_df,
                    feature_importance_dict=feature_importance_dict if feature_importance_dict else None,
                    save_dir=temp_dir,
                    cv_results=cv_results,
                    horizon_results=horizon_results,
                )
                
                logger.info(f"Generated {len(saved_files)} visualization files: {list(saved_files.keys())}")
                
                # Log each visualization to MLflow
                for viz_name, file_path in saved_files.items():
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path, "visualizations")
                        logger.info(f"Logged visualization: {viz_name} from {file_path}")
                    else:
                        logger.warning(f"Visualization file not found: {file_path}")
                
                # Also create a combined HTML report
                self._create_combined_html_report(saved_files, temp_dir)
                
                # Log the combined report
                combined_report = os.path.join(temp_dir, 'model_comparison_report.html')
                if os.path.exists(combined_report):
                    mlflow.log_artifact(combined_report, "reports")
                    logger.info("Logged combined HTML report")
                    
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            # Don't fail the entire training if visualization fails
    
    def _create_combined_html_report(self, saved_files: Dict[str, str], save_dir: str) -> None:
        """Create a combined HTML report with all visualizations"""
        import os
        from datetime import datetime
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                h1, h2 {
                    color: #333;
                }
                .section {
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .timestamp {
                    color: #666;
                    font-size: 14px;
                }
                iframe {
                    width: 100%;
                    height: 800px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    margin-top: 10px;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 4px;
                    margin-top: 10px;
                }
            </style>
        </head>
        <body>
            <h1>Sales Forecast Model Comparison Report</h1>
            <p class="timestamp">Generated on: {timestamp}</p>
        """
        
        html_content = html_content.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Add each visualization section
        sections = [
            ('metrics_comparison', 'Model Performance Metrics (MAE / RMSE / WAPE / Bias)'),
            ('metrics_comparison_table', 'Model Comparison Table'),
            ('predictions_comparison', 'Predictions Comparison'),
            ('residuals_analysis', 'Residuals Analysis'),
            ('error_distribution', 'Error Distribution'),
            ('feature_importance', 'Feature Importance'),
            ('rolling_origin_cv', 'Rolling-Origin Cross-Validation'),
            ('horizon_accuracy', 'Forecast Accuracy by Horizon'),
            ('summary', 'Summary Statistics')
        ]
        
        for key, title in sections:
            if key in saved_files:
                file_path = saved_files[key]
                if not str(file_path).lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                html_content += f'<div class="section"><h2>{title}</h2>'
                
                # All image files are PNG - base64 encode them
                import base64
                with open(file_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                html_content += f'<img src="data:image/png;base64,{img_data}" alt="{title}">'
                
                html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        # Save the combined report
        with open(os.path.join(save_dir, 'model_comparison_report.html'), 'w') as f:
            f.write(html_content)
    
    def save_artifacts(self):
        # Save scalers and encoders
        joblib.dump(self.scalers, '/tmp/scalers.pkl')
        joblib.dump(self.encoders, '/tmp/encoders.pkl')
        joblib.dump(self.feature_cols, '/tmp/feature_cols.pkl')

        # Save individual models in the expected format
        import os
        os.makedirs('/tmp/models/xgboost', exist_ok=True)
        os.makedirs('/tmp/models/lightgbm', exist_ok=True)
        os.makedirs('/tmp/models/ensemble', exist_ok=True)

        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], '/tmp/models/xgboost/xgboost_model.pkl')

        if 'lightgbm' in self.models:
            joblib.dump(self.models['lightgbm'], '/tmp/models/lightgbm/lightgbm_model.pkl')

        if 'ensemble' in self.models:
            joblib.dump(self.models['ensemble'], '/tmp/models/ensemble/ensemble_model.pkl')

        for model_name in CLASSICAL_TS_MODELS:
            if model_name in self.models:
                model_dir = f'/tmp/models/{model_name}'
                os.makedirs(model_dir, exist_ok=True)
                joblib.dump(self.models[model_name], f'{model_dir}/{model_name}_model.pkl')

        self.mlflow_manager.log_artifacts('/tmp/')

        logger.info("Artifacts saved successfully")