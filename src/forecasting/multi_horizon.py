"""
Multi-horizon forecasting (CDC continuation Phase 5/6).

Architecture choice: ONE MODEL PER HORIZON (documented decision).
Rationale: horizons differ fundamentally in which features are legitimately
available at issue time (intra-day can use very recent production lags;
J+1/J+2/J+3 cannot - only forecast weather, capacity, solar geometry, and
calendar are available that far ahead operationally). A single multi-output
model would force one shared feature set across horizons, which would either
starve intra-day of useful recent-lag information or leak lag information
into J+1-J+3. Separate models let each horizon use exactly the features that
would be genuinely available operationally at its own issue time.

For each horizon h in {15min intra-day, J+1, J+2, J+3}, builds an explicit
supervised frame:
    forecast_issue_time, target_timestamp, horizon, district_id,
    <horizon-appropriate features>, target = production at target_timestamp
using src.forecasting.backtesting.shift_to_horizon for the label, and fits +
backtests a LightGBM regressor per horizon.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.forecasting.backtesting import shift_to_horizon, HORIZON_STEPS_15MIN

# Features considered genuinely available at issue time for each horizon.
# Intra-day: recent production lags ARE available (last observation is real).
# J+1/J+2/J+3: NO production lags - only forecast-weather-equivalent inputs
# (in this repo, the same weather columns stand in for "forecast weather",
# since WeatherProvider.get_forecast_weather returns the same schema).
INTRADAY_FEATURES = [
    "capacity_mw", "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m",
    "solar_elevation_deg", "solar_azimuth_deg", "clearsky_ghi_wm2", "daylight_flag",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "day_of_week", "month",
    "pv_production_mw_proxy_lag_1", "pv_production_mw_proxy_lag_4",
    "pv_production_mw_proxy_rollmean_4",
    "pv_production_mw_reference_lag_1", "pv_production_mw_reference_lag_4",
    "pv_production_mw_reference_rollmean_4",
]

DAY_AHEAD_FEATURES = [  # used for J+1, J+2, J+3 alike - no production lags
    "capacity_mw", "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m",
    "solar_elevation_deg", "solar_azimuth_deg", "clearsky_ghi_wm2", "daylight_flag",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "day_of_week", "month",
]

HORIZONS = {
    "intraday_15min": {"steps": HORIZON_STEPS_15MIN["15min"], "features": INTRADAY_FEATURES},
    "J+1": {"steps": HORIZON_STEPS_15MIN["J+1"], "features": DAY_AHEAD_FEATURES},
    "J+2": {"steps": HORIZON_STEPS_15MIN["J+2"], "features": DAY_AHEAD_FEATURES},
    "J+3": {"steps": HORIZON_STEPS_15MIN["J+3"], "features": DAY_AHEAD_FEATURES},
}


@dataclass
class HorizonModel:
    horizon_name: str
    feature_cols: list
    model: object = None

    def build_supervised_frame(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        steps = HORIZONS[self.horizon_name]["steps"]
        out = df.copy()
        out["target_at_horizon"] = shift_to_horizon(out, target_col, steps)
        out["horizon"] = self.horizon_name
        out["forecast_issue_time"] = out["timestamp"]
        out["target_timestamp"] = out["timestamp"] + pd.to_timedelta(steps * 15, unit="min")
        cols = [c for c in self.feature_cols if c in out.columns]
        self.feature_cols = cols
        return out

    def fit(self, train_df: pd.DataFrame):
        import lightgbm as lgb
        X = train_df[self.feature_cols]
        y = train_df["target_at_horizon"]
        valid = y.notna() & X.notna().all(axis=1)
        self.model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
        )
        self.model.fit(X[valid], y[valid])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_cols]
        pred = np.full(len(df), np.nan)
        valid = X.notna().all(axis=1)
        pred[valid.to_numpy()] = self.model.predict(X[valid])
        return pd.Series(pred, index=df.index).clip(lower=0)
