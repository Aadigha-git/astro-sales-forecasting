"""
Model visualization and comparison module for sales forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Removed plotly imports - using matplotlib only
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Create comprehensive visualizations for model comparison and analysis"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize the visualizer with plotting style"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')
        
        self.colors = {
            'xgboost': '#FF6B6B',
            'lightgbm': '#4ECDC4',
            'prophet': '#45B7D1',
            'seasonal_naive': '#F39C12',
            'holt_winters': '#9B59B6',
            'sarimax': '#E67E22',
            'ensemble': '#96CEB4',
            'actual': '#2C3E50'
        }
        
    def create_metrics_comparison_chart(self, metrics_dict: Dict[str, Dict[str, float]], 
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Create a comparison chart for MAE, RMSE, WAPE, and forecast bias."""
        
        # Prepare data
        models = list(metrics_dict.keys())
        
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Metrics Comparison', fontsize=16)
        
        # MAE / RMSE / WAPE: lower is better; bias: closest to zero is best
        metrics_to_plot = [
            ('mae', 'MAE', 'lower', axes[0, 0]),
            ('rmse', 'RMSE', 'lower', axes[0, 1]),
            ('wape', 'WAPE (%)', 'lower', axes[1, 0]),
            ('bias', 'Forecast Bias (pred − actual)', 'closest_to_zero', axes[1, 1]),
        ]
        
        for metric, title, best_mode, ax in metrics_to_plot:
            values = [float(metrics_dict[model].get(metric, 0) or 0) for model in models]
            colors = [self.colors.get(model.lower(), '#95A5A6') for model in models]
            
            bars = ax.bar(models, values, color=colors, alpha=0.7)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va=va, fontsize=8)
            
            if best_mode == 'lower':
                best_idx = int(np.argmin(values))
            else:  # closest_to_zero
                best_idx = int(np.argmin(np.abs(values)))
            
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(3)
            
            ax.set_title(f'{title} Comparison')
            ax.set_ylabel(title)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, alpha=0.3)
            if metric == 'bias':
                ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.6)
                pad = max(abs(v) for v in values) * 0.2 if values else 1.0
                pad = pad if pad > 0 else 1.0
                ax.set_ylim(min(values) - pad, max(values) + pad)
            else:
                ymax = max(values) if values else 1.0
                ax.set_ylim(min(0, min(values)), ymax * 1.15 if ymax > 0 else 1.0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved metrics comparison chart to {save_path}")
        
        return fig

    def create_metrics_comparison_table(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Build a ranked comparison table for MAE, RMSE, WAPE, and bias.

        Saves a CSV (if csv_path) and a rendered table figure (if save_path).
        """
        metrics = metrics or ["mae", "rmse", "wape", "bias"]
        rows = []
        for model_name, model_metrics in metrics_dict.items():
            row = {"model": model_name}
            for m in metrics:
                row[m] = float(model_metrics.get(m, np.nan))
            rows.append(row)

        table_df = pd.DataFrame(rows)
        if not table_df.empty:
            # Rank by MAE then RMSE (primary accuracy); bias abs as tie-breaker
            table_df = table_df.sort_values(
                by=["mae", "rmse", "wape"], ascending=True, na_position="last"
            ).reset_index(drop=True)
            table_df.insert(0, "rank", np.arange(1, len(table_df) + 1))

            # Highlight helpers
            table_df["abs_bias"] = table_df["bias"].abs()
            best = {
                "mae": table_df["mae"].idxmin() if table_df["mae"].notna().any() else None,
                "rmse": table_df["rmse"].idxmin() if table_df["rmse"].notna().any() else None,
                "wape": table_df["wape"].idxmin() if table_df["wape"].notna().any() else None,
                "bias": table_df["abs_bias"].idxmin() if table_df["abs_bias"].notna().any() else None,
            }
        else:
            best = {}

        display_cols = ["rank", "model", "mae", "rmse", "wape", "bias"]
        display_df = table_df[display_cols].copy() if not table_df.empty else table_df
        for col in ["mae", "rmse", "wape", "bias"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )

        if csv_path and not table_df.empty:
            export_df = table_df[["rank", "model", "mae", "rmse", "wape", "bias"]].copy()
            export_df.to_csv(csv_path, index=False)
            logger.info(f"Saved metrics comparison table CSV to {csv_path}")

        fig, ax = plt.subplots(figsize=(11, max(2.5, 0.55 * max(len(display_df), 1) + 1.5)))
        ax.axis("off")
        ax.set_title(
            "Model Comparison — MAE / RMSE / WAPE / Bias\n"
            "(lower MAE/RMSE/WAPE better; bias closer to 0 better; +bias = over-forecast)",
            fontsize=12,
            pad=12,
        )

        if display_df.empty:
            ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        else:
            table = ax.table(
                cellText=display_df.values,
                colLabels=["Rank", "Model", "MAE", "RMSE", "WAPE (%)", "Bias"],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.15, 1.4)

            # Style header
            for j in range(len(display_df.columns)):
                table[(0, j)].set_facecolor("#2C3E50")
                table[(0, j)].set_text_props(color="white", weight="bold")

            # Color best cells lightly
            col_index = {"mae": 2, "rmse": 3, "wape": 4, "bias": 5}
            for metric_name, col_idx in col_index.items():
                row_idx = best.get(metric_name)
                if row_idx is None or pd.isna(row_idx):
                    continue
                # +1 for header row in matplotlib table
                table[(int(row_idx) + 1, col_idx)].set_facecolor("#D5F5E3")

            for i in range(1, len(display_df) + 1):
                model_name = str(table_df.loc[i - 1, "model"]).lower()
                color = self.colors.get(model_name, "#ECF0F1")
                table[(i, 1)].set_facecolor(color)
                table[(i, 1)].set_alpha(0.35)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved metrics comparison table to {save_path}")

        return table_df.drop(columns=["abs_bias"], errors="ignore"), fig

    def create_predictions_comparison_chart(self, predictions_dict: Dict[str, pd.DataFrame],
                                          actual_data: pd.DataFrame,
                                          date_col: str = 'date',
                                          target_col: str = 'sales',
                                          save_path: Optional[str] = None) -> plt.Figure:
        """Create time series comparison of model predictions"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Add actual data
        ax.plot(actual_data[date_col], actual_data[target_col], 
                color=self.colors['actual'], linewidth=3, 
                label='Actual', alpha=0.8)
        
        # Add predictions for each model
        for model_name, pred_df in predictions_dict.items():
            color = self.colors.get(model_name.lower(), '#95A5A6')
            
            ax.plot(pred_df[date_col], pred_df['prediction'],
                   color=color, linewidth=2, 
                   label=f'{model_name} Prediction', alpha=0.7)
            
            # Add confidence intervals if available
            if 'prediction_lower' in pred_df.columns and 'prediction_upper' in pred_df.columns:
                ax.fill_between(pred_df[date_col], 
                               pred_df['prediction_lower'], 
                               pred_df['prediction_upper'],
                               color=color, alpha=0.1)
        
        ax.set_title('Model Predictions Comparison', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(target_col.capitalize(), fontsize=12)
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved predictions comparison chart to {save_path}")
        
        return fig
    
    def create_residuals_analysis(self, predictions_dict: Dict[str, pd.DataFrame],
                                actual_data: pd.DataFrame,
                                target_col: str = 'sales',
                                save_path: Optional[str] = None) -> plt.Figure:
        """Create residuals analysis plots"""
        
        # Calculate residuals for each model
        residuals_data = {}
        merged_data = {}  # Keep track of merged dataframes
        for model_name, pred_df in predictions_dict.items():
            # Merge predictions with actual data
            merged = pd.merge(
                actual_data[['date', target_col]], 
                pred_df[['date', 'prediction']], 
                on='date',
                how='inner'
            )
            residuals_data[model_name] = merged[target_col] - merged['prediction']
            merged_data[model_name] = merged  # Store the merged dataframe
        
        # Create matplotlib subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Residuals Analysis', fontsize=16)
        
        # 1. Box plot of residuals
        ax1 = axes[0, 0]
        box_data = [residuals for residuals in residuals_data.values()]
        box_colors = [self.colors.get(model.lower(), '#95A5A6') for model in residuals_data.keys()]
        
        bp = ax1.boxplot(box_data, labels=list(residuals_data.keys()), patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Residuals Distribution')
        ax1.set_ylabel('Residuals')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Residuals vs Predicted (for first model)
        ax2 = axes[0, 1]
        first_model = list(predictions_dict.keys())[0]
        first_pred = predictions_dict[first_model]
        first_residuals = residuals_data[first_model]
        
        # Ensure we have matching lengths
        min_len = min(len(first_pred), len(first_residuals))
        pred_values = first_pred['prediction'].values[:min_len]
        resid_values = first_residuals.values[:min_len]
        
        ax2.scatter(pred_values, resid_values,
                   color=self.colors.get(first_model.lower(), '#95A5A6'),
                   alpha=0.6, s=30)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_title(f'Residuals vs Predicted ({first_model})')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals over time
        ax3 = axes[1, 0]
        for model_name in residuals_data.keys():
            if model_name in merged_data:
                # Use the dates from merged data to ensure alignment
                dates = merged_data[model_name]['date']
                residuals = residuals_data[model_name]
                
                ax3.plot(dates, residuals,
                        color=self.colors.get(model_name.lower(), '#95A5A6'),
                        label=model_name, alpha=0.7)
            else:
                # Fallback for backward compatibility
                residuals = residuals_data[model_name]
                pred_df = predictions_dict[model_name]
                min_len = min(len(pred_df), len(residuals))
                dates = pred_df['date'].iloc[:min_len]
                resid_values = residuals.iloc[:min_len] if hasattr(residuals, 'iloc') else residuals[:min_len]
                
                ax3.plot(dates, resid_values,
                        color=self.colors.get(model_name.lower(), '#95A5A6'),
                        label=model_name, alpha=0.7)
        
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.set_title('Residuals Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Residuals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        
        # 4. Q-Q plot (for first model)
        ax4 = axes[1, 1]
        from scipy import stats
        # Use the residuals array directly
        resid_array = first_residuals.values if hasattr(first_residuals, 'values') else first_residuals
        theoretical_quantiles = stats.probplot(resid_array, dist="norm", fit=False)[0]
        
        ax4.scatter(theoretical_quantiles, sorted(resid_array),
                   color=self.colors.get(first_model.lower(), '#95A5A6'),
                   alpha=0.6)
        
        # Add diagonal reference line
        min_val = min(theoretical_quantiles.min(), resid_array.min())
        max_val = max(theoretical_quantiles.max(), resid_array.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax4.set_title(f'Q-Q Plot ({first_model})')
        ax4.set_xlabel('Theoretical Quantiles')
        ax4.set_ylabel('Sample Quantiles')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved residuals analysis chart to {save_path}")
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance_dict: Dict[str, pd.DataFrame],
                                      top_n: int = 20,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Create feature importance comparison chart"""
        
        n_models = len(feature_importance_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8), sharey=False)
        
        # Handle single model case
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importance_df) in enumerate(feature_importance_dict.items()):
            ax = axes[idx]
            
            # Get top N features
            top_features = importance_df.nlargest(top_n, 'importance')
            
            # Create horizontal bar chart
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_features['importance'], 
                   color=self.colors.get(model_name.lower(), '#95A5A6'),
                   alpha=0.7)
            
            # Add value labels
            for i, v in enumerate(top_features['importance']):
                ax.text(v, i, f' {v:.3f}', va='center')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name} - Top {top_n} Features')
            ax.grid(True, alpha=0.3, axis='x')
            
            if idx == 0:
                ax.set_ylabel('Features')
        
        fig.suptitle(f'Top {top_n} Feature Importance by Model', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved feature importance chart to {save_path}")
        
        return fig
    
    def create_error_distribution_chart(self, predictions_dict: Dict[str, pd.DataFrame],
                                      actual_data: pd.DataFrame,
                                      target_col: str = 'sales',
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Create error distribution visualization"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, pred_df in predictions_dict.items():
            # Merge and calculate errors
            merged = pd.merge(
                actual_data[['date', target_col]], 
                pred_df[['date', 'prediction']], 
                on='date',
                how='inner'
            )
            errors = (merged[target_col] - merged['prediction']).abs()
            
            # Create histogram
            ax.hist(errors, bins=50, alpha=0.7,
                   color=self.colors.get(model_name.lower(), '#95A5A6'),
                   label=model_name, density=True)
        
        ax.set_title('Absolute Error Distribution by Model', fontsize=16)
        ax.set_xlabel('Absolute Error', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved error distribution chart to {save_path}")
        
        return fig
    
    def create_comprehensive_report(self, metrics_dict: Dict[str, Dict[str, float]],
                                  predictions_dict: Dict[str, pd.DataFrame],
                                  actual_data: pd.DataFrame,
                                  feature_importance_dict: Optional[Dict[str, pd.DataFrame]] = None,
                                  save_dir: str = '/tmp/model_comparison_charts',
                                  cv_results: Optional[Dict[str, Any]] = None,
                                  horizon_results: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate all comparison charts and save them"""
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = {}
        
        # 1. Metrics comparison (MAE / RMSE / WAPE / bias)
        self.create_metrics_comparison_chart(
            metrics_dict,
            save_path=os.path.join(save_dir, 'metrics_comparison.png')
        )
        saved_files['metrics_comparison'] = os.path.join(save_dir, 'metrics_comparison.png')

        # 1b. Metrics comparison table
        table_png = os.path.join(save_dir, 'metrics_comparison_table.png')
        table_csv = os.path.join(save_dir, 'metrics_comparison_table.csv')
        self.create_metrics_comparison_table(
            metrics_dict,
            save_path=table_png,
            csv_path=table_csv,
            metrics=["mae", "rmse", "wape", "bias"],
        )
        saved_files['metrics_comparison_table'] = table_png
        saved_files['metrics_comparison_table_csv'] = table_csv
        
        # 2. Predictions comparison
        self.create_predictions_comparison_chart(
            predictions_dict,
            actual_data,
            save_path=os.path.join(save_dir, 'predictions_comparison.png')
        )
        saved_files['predictions_comparison'] = os.path.join(save_dir, 'predictions_comparison.png')
        
        # 3. Residuals analysis
        self.create_residuals_analysis(
            predictions_dict,
            actual_data,
            save_path=os.path.join(save_dir, 'residuals_analysis.png')
        )
        saved_files['residuals_analysis'] = os.path.join(save_dir, 'residuals_analysis.png')
        
        # 4. Error distribution
        self.create_error_distribution_chart(
            predictions_dict,
            actual_data,
            save_path=os.path.join(save_dir, 'error_distribution.png')
        )
        saved_files['error_distribution'] = os.path.join(save_dir, 'error_distribution.png')
        
        # 5. Feature importance (if available)
        if feature_importance_dict:
            self.create_feature_importance_chart(
                feature_importance_dict,
                save_path=os.path.join(save_dir, 'feature_importance.png')
            )
            saved_files['feature_importance'] = os.path.join(save_dir, 'feature_importance.png')

        # 6. Rolling-origin CV aggregates (if available)
        if cv_results and cv_results.get("models"):
            cv_path = os.path.join(save_dir, 'rolling_origin_cv.png')
            self.create_rolling_origin_cv_chart(cv_results, save_path=cv_path)
            saved_files['rolling_origin_cv'] = cv_path

        # 7. Accuracy / bias vs forecast horizon
        if horizon_results and horizon_results.get("accuracy_by_horizon"):
            h_path = os.path.join(save_dir, 'horizon_accuracy.png')
            self.create_horizon_accuracy_chart(horizon_results, save_path=h_path)
            saved_files['horizon_accuracy'] = h_path
        
        # Create summary matplotlib figure
        self._create_summary_figure(metrics_dict, save_dir)
        saved_files['summary'] = os.path.join(save_dir, 'model_comparison_summary.png')
        
        logger.info(f"Generated {len(saved_files)} visualization files in {save_dir}")
        return saved_files

    def create_horizon_accuracy_chart(
        self,
        horizon_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Line charts showing how forecast accuracy and bias change with horizon.

        X-axis: forecast horizon (days). Y-axis: RMSE / MAE / WAPE / bias.
        One line per model.
        """
        rows = horizon_results.get("accuracy_by_horizon") or []
        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        horizons_cfg = horizon_results.get("config", {}).get("horizons", [])
        fig.suptitle(
            f"Forecast Accuracy vs Horizon ({', '.join(str(h) + 'd' for h in horizons_cfg) or 'multi-horizon'})",
            fontsize=14,
        )

        specs = [
            ("rmse", "RMSE (lower better)", axes[0, 0], False),
            ("mae", "MAE (lower better)", axes[0, 1], False),
            ("wape", "WAPE % (lower better)", axes[1, 0], False),
            ("bias", "Forecast Bias (closer to 0 better)", axes[1, 1], True),
        ]

        if df.empty:
            for _, _, ax, _ in specs:
                ax.text(0.5, 0.5, "No horizon metrics", ha="center", va="center")
                ax.axis("off")
        else:
            for metric, ylabel, ax, signed in specs:
                for model_name, g in df.groupby("model"):
                    g = g.sort_values("horizon")
                    color = self.colors.get(str(model_name).lower(), "#95A5A6")
                    y = g[metric].astype(float).values
                    x = g["horizon"].astype(int).values
                    yerr = g[f"{metric}_std"] if f"{metric}_std" in g.columns else None
                    if yerr is not None:
                        ax.errorbar(
                            x,
                            y,
                            yerr=yerr.astype(float).values,
                            marker="o",
                            color=color,
                            label=model_name,
                            linewidth=2,
                            capsize=3,
                        )
                    else:
                        ax.plot(x, y, marker="o", color=color, label=model_name, linewidth=2)
                ax.set_xlabel("Forecast horizon (days)")
                ax.set_ylabel(ylabel)
                ax.set_title(ylabel)
                ax.grid(True, alpha=0.3)
                if signed:
                    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
                # Integer-like ticks at evaluated horizons
                xticks = sorted(df["horizon"].unique())
                ax.set_xticks(xticks)
                ax.legend(fontsize=8, loc="best")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved horizon accuracy chart to {save_path}")
        return fig

    def create_rolling_origin_cv_chart(
        self, cv_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> plt.Figure:
        """Bar charts of mean±std for MAE, RMSE, WAPE, and bias across CV folds."""
        models = []
        series = {
            "mae": ([], []),
            "rmse": ([], []),
            "wape": ([], []),
            "bias": ([], []),
        }

        for model_name, payload in cv_results.get("models", {}).items():
            agg = payload.get("aggregated", {})
            if "rmse_mean" not in agg and "mae_mean" not in agg:
                continue
            models.append(model_name)
            for metric in series:
                series[metric][0].append(agg.get(f"{metric}_mean", 0.0))
                series[metric][1].append(agg.get(f"{metric}_std", 0.0))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        n_folds = cv_results.get("config", {}).get("n_folds", "?")
        horizon = cv_results.get("config", {}).get("horizon", "?")
        fig.suptitle(
            f"Rolling-Origin CV — MAE / RMSE / WAPE / Bias "
            f"(n_folds={n_folds}, horizon={horizon})",
            fontsize=14,
        )

        plot_specs = [
            ("mae", "MAE", axes[0, 0], False),
            ("rmse", "RMSE", axes[0, 1], False),
            ("wape", "WAPE (%)", axes[1, 0], False),
            ("bias", "Forecast Bias", axes[1, 1], True),
        ]

        if models:
            x = np.arange(len(models))
            colors = [self.colors.get(m.lower(), "#95A5A6") for m in models]
            for metric, ylabel, ax, signed in plot_specs:
                means, stds = series[metric]
                ax.bar(x, means, yerr=stds, color=colors, alpha=0.75, capsize=4)
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=30, ha="right")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{ylabel} mean ± std")
                ax.grid(True, alpha=0.3)
                if signed:
                    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        else:
            for _, _, ax, _ in plot_specs:
                ax.text(0.5, 0.5, "No CV metrics", ha="center", va="center")
                ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved rolling-origin CV chart to {save_path}")
        return fig

    def _create_summary_figure(self, metrics_dict: Dict[str, Dict[str, float]], 
                              save_dir: str) -> None:
        """Create a summary figure using matplotlib"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Summary (MAE / RMSE / WAPE / Bias)', fontsize=16)
        
        models = list(metrics_dict.keys())
        metric_specs = [
            ('mae', 'MAE', 'lower'),
            ('rmse', 'RMSE', 'lower'),
            ('wape', 'WAPE (%)', 'lower'),
            ('bias', 'Forecast Bias', 'closest_to_zero'),
        ]
        
        for ax, (metric, label, best_mode) in zip(axes.flat, metric_specs):
            values = [float(metrics_dict[model].get(metric, 0) or 0) for model in models]
            colors = [self.colors.get(model.lower(), '#95A5A6') for model in models]
            
            bars = ax.bar(models, values, color=colors, alpha=0.7)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va=va, fontsize=8)
            
            ax.set_title(f'{label} Comparison')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, alpha=0.3)
            
            if best_mode == 'closest_to_zero':
                best_idx = int(np.argmin(np.abs(values))) if values else 0
                ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.6)
            else:
                best_idx = int(np.argmin(values)) if values else 0
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def generate_model_comparison_report(mlflow_manager, run_id: str, 
                                   test_data: pd.DataFrame) -> Dict[str, str]:
    """Helper function to generate comparison report from MLflow run"""
    
    visualizer = ModelVisualizer()
    
    # Get run data from MLflow
    import mlflow
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    
    # Extract metrics
    metrics_dict = {}
    for model in [
        'xgboost', 'lightgbm', 'prophet',
        'seasonal_naive', 'holt_winters', 'sarimax', 'ensemble'
    ]:
        model_metrics = {}
        for metric in ['rmse', 'mae', 'mape', 'wape', 'bias', 'r2']:
            metric_key = f"{model}_{metric}"
            if metric_key in run.data.metrics:
                model_metrics[metric] = run.data.metrics[metric_key]
        if model_metrics:
            metrics_dict[model] = model_metrics
    
    # Generate dummy predictions for visualization
    # In real scenario, load actual predictions from artifacts
    predictions_dict = {}
    for model in metrics_dict.keys():
        pred_df = test_data[['date']].copy()
        # Add some noise to create different predictions
        noise = np.random.normal(0, 5, len(test_data))
        pred_df['prediction'] = test_data['sales'] + noise
        predictions_dict[model] = pred_df
    
    # Generate visualizations
    saved_files = visualizer.create_comprehensive_report(
        metrics_dict,
        predictions_dict,
        test_data
    )
    
    # Log visualizations to MLflow
    for name, path in saved_files.items():
        mlflow.log_artifact(path, f"visualizations/{name}")
    
    return saved_files