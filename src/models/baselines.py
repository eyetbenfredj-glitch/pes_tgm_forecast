"""
Baseline forecast models (CDC section 10 & continuation section 10/11).
All operate on a per-district long dataframe with columns:
    district_id, timestamp, <target_col>, capacity_mw, daylight_flag, ...

Baselines implemented:
  1. Persistence            - forecast(t+h) = observed(t)
  2. Seasonal persistence    - forecast(t+h) = observed(t+h-1 day) [or -1 week]
  3. Physics / clear-sky     - delegates to src.physics.pv_model (already real physics)
  4. ML baseline (LightGBM)  - gradient boosting regressor on engineered features

Each baseline exposes a common `.predict(df) -> pd.Series` interface so they
are interchangeable in scripts/05_train_and_backtest.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PersistenceBaseline:
    """forecast(t) = value observed `lag_steps` steps earlier for the same district."""
    target_col: str
    lag_steps: int = 1  # e.g. 1 step = last observed value at the forecast horizon's step size

    def predict(self, df: pd.DataFrame) -> pd.Series:
        df = df.sort_values(["district_id", "timestamp"])
        return df.groupby("district_id")[self.target_col].shift(self.lag_steps)


@dataclass
class SeasonalPersistenceBaseline:
    """forecast(t) = value observed at the same time-of-day `season_steps` steps earlier
    (e.g. 96 steps = same time yesterday at 15-min resolution)."""
    target_col: str
    season_steps: int = 96

    def predict(self, df: pd.DataFrame) -> pd.Series:
        df = df.sort_values(["district_id", "timestamp"])
        return df.groupby("district_id")[self.target_col].shift(self.season_steps)


@dataclass
class PhysicsBaseline:
    """Wraps src.physics.pv_model.physics_pv_output_mw - a genuine physical
    computation (pvlib clear-sky + NOCT temperature derating), not a fitted model."""
    performance_ratio: float = 0.80
    noct_c: float = 45.0
    temp_coeff_pct_per_c: float = -0.40

    def predict(self, df: pd.DataFrame) -> pd.Series:
        from src.physics.pv_model import physics_pv_output_mw
        return pd.Series(
            physics_pv_output_mw(
                capacity_mw=df["capacity_mw"].to_numpy(),
                ghi_wm2=df["shortwave_radiation"].to_numpy(),
                clearsky_ghi_wm2=df["clearsky_ghi_wm2"].to_numpy(),
                ambient_temp_c=df["temperature_2m"].to_numpy(),
                performance_ratio=self.performance_ratio,
                noct_c=self.noct_c,
                temp_coeff_pct_per_c=self.temp_coeff_pct_per_c,
            ),
            index=df.index,
        )


DEFAULT_ML_FEATURES = [
    "capacity_mw", "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m",
    "solar_elevation_deg", "solar_azimuth_deg", "clearsky_ghi_wm2", "daylight_flag",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "day_of_week", "month",
]


@dataclass
class LightGBMBaseline:
    """Gradient-boosting regressor baseline. Trained via .fit(train_df, target_col),
    predicts via .predict(df). Features restricted to DEFAULT_ML_FEATURES (extend
    with lag features only if the caller guarantees they were built leakage-safely)."""
    feature_cols: list = field(default_factory=lambda: list(DEFAULT_ML_FEATURES))
    params: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
    ))
    model: object = None

    def fit(self, df: pd.DataFrame, target_col: str):
        import lightgbm as lgb
        cols = [c for c in self.feature_cols if c in df.columns]
        self.feature_cols = cols
        X = df[cols]
        y = df[target_col]
        valid = y.notna() & X.notna().all(axis=1)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X[valid], y[valid])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_cols]
        pred = np.full(len(df), np.nan)
        valid = X.notna().all(axis=1)
        pred[valid.to_numpy()] = self.model.predict(X[valid])
        return pd.Series(pred, index=df.index).clip(lower=0)


@dataclass
class XGBoostBaseline:
    """XGBoost regressor baseline."""
    feature_cols: list = field(default_factory=lambda: list(DEFAULT_ML_FEATURES))
    params: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        reg_alpha=0, reg_lambda=1, gamma=0, random_state=42, n_jobs=-1,
    ))
    model: object = None

    def fit(self, df: pd.DataFrame, target_col: str):
        import xgboost as xgb
        cols = [c for c in self.feature_cols if c in df.columns]
        self.feature_cols = cols
        X = df[cols]
        y = df[target_col]
        valid = y.notna() & X.notna().all(axis=1)
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X[valid], y[valid])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_cols]
        pred = np.full(len(df), np.nan)
        valid = X.notna().all(axis=1)
        pred[valid.to_numpy()] = self.model.predict(X[valid])
        return pd.Series(pred, index=df.index).clip(lower=0)
