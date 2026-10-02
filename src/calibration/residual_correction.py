"""
Forecast error correction engine (CDC continuation section 19/20).

Pipeline:
    raw_forecast -> observed -> residual = observed - raw_forecast
                 -> residual model (features: hour, solar elevation, cloud
                    cover, district, recent residual history)
                 -> corrected_forecast = raw_forecast + residual_model(features)

Two strategies implemented:
  1. RollingBiasCorrector   - simple per-district rolling mean bias (fast,
                              always available, good fallback per CDC L2 in
                              the fallback ladder).
  2. ResidualMLCorrector    - LightGBM regressor predicting the residual from
                              contextual features, refit periodically
                              (scripts/06_update_forecast_model.py).

Both expose .fit(history_df) / .correct(forecast_df) -> corrected Series, so
they are interchangeable, and both are exercised end-to-end in
tests/test_correction.py with a synthetic-but-labelled experiment showing
raw vs corrected error actually decreasing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RollingBiasCorrector:
    window: int = 96 * 7  # 1 week of 15-min steps
    _bias_by_district: dict = field(default_factory=dict)

    def fit(self, history_df: pd.DataFrame, actual_col: str, forecast_col: str):
        h = history_df.copy()
        h["residual"] = h[actual_col] - h[forecast_col]
        for did, g in h.sort_values("timestamp").groupby("district_id"):
            self._bias_by_district[did] = float(g["residual"].tail(self.window).mean())
        return self

    def correct(self, forecast_df: pd.DataFrame, forecast_col: str) -> pd.Series:
        bias = forecast_df["district_id"].map(self._bias_by_district).fillna(0.0)
        corrected = forecast_df[forecast_col] + bias
        # physical bound: never push below 0 or above capacity if available
        if "capacity_mw" in forecast_df.columns:
            corrected = corrected.clip(lower=0, upper=forecast_df["capacity_mw"])
        else:
            corrected = corrected.clip(lower=0)
        return corrected


RESIDUAL_FEATURES = [
    "solar_elevation_deg", "cloud_cover", "hour_sin", "hour_cos", "capacity_mw",
]


@dataclass
class ResidualMLCorrector:
    feature_cols: list = field(default_factory=lambda: list(RESIDUAL_FEATURES))
    model: object = None

    def fit(self, history_df: pd.DataFrame, actual_col: str, forecast_col: str):
        import lightgbm as lgb
        h = history_df.copy()
        h["residual"] = h[actual_col] - h[forecast_col]
        cols = [c for c in self.feature_cols if c in h.columns]
        self.feature_cols = cols
        X, y = h[cols], h["residual"]
        valid = y.notna() & X.notna().all(axis=1)
        self.model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            random_state=42, verbosity=-1,
        )
        self.model.fit(X[valid], y[valid])
        return self

    def correct(self, forecast_df: pd.DataFrame, forecast_col: str) -> pd.Series:
        X = forecast_df[self.feature_cols]
        valid = X.notna().all(axis=1)
        residual_pred = np.zeros(len(forecast_df))
        residual_pred[valid.to_numpy()] = self.model.predict(X[valid])
        corrected = forecast_df[forecast_col] + residual_pred
        if "capacity_mw" in forecast_df.columns:
            corrected = corrected.clip(lower=0, upper=forecast_df["capacity_mw"])
        else:
            corrected = corrected.clip(lower=0)
        return corrected


def correction_experiment_report(y_true: pd.Series, raw_forecast: pd.Series,
                                  corrected_forecast: pd.Series, capacity: pd.Series) -> dict:
    """raw vs corrected error, for the demo/report (CDC section 19)."""
    from src.forecasting.backtesting import evaluate
    return {
        "raw": evaluate(y_true, raw_forecast, capacity),
        "corrected": evaluate(y_true, corrected_forecast, capacity),
    }
