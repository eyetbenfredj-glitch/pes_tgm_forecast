"""
Physical constraints for PV forecasting.

Applies physically justified post-processing to ML forecasts:
1. P_pv(t) = 0 if solar_elevation <= 0 (Nighttime zeroing)
2. 0 <= P_pv(t) <= Installed_Capacity (Bounding)

Returns the constrained forecast and measures the impact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_physical_constraints(
    df: pd.DataFrame,
    forecast_col: str,
    capacity_col: str = "capacity_mw",
    elevation_col: str = "solar_elevation_deg",
) -> pd.DataFrame:
    """
    Applies physical constraints to a forecast column in-place and returns
    a new Series with the constrained values.
    """
    forecast = df[forecast_col].copy()
    
    # Lower bound: 0
    forecast = forecast.clip(lower=0)
    
    # Upper bound: installed capacity
    if capacity_col in df.columns:
        forecast = forecast.clip(upper=df[capacity_col])
        
    # Nighttime zeroing
    if elevation_col in df.columns:
        night_mask = df[elevation_col] <= 0
        forecast.loc[night_mask] = 0.0
        
    return forecast


def measure_constraint_impact(
    df: pd.DataFrame,
    original_forecast: pd.Series,
    constrained_forecast: pd.Series,
) -> dict:
    """
    Measures how many rows were modified by the physical constraints.
    """
    diff = np.abs(original_forecast - constrained_forecast)
    modified = (diff > 1e-6).sum()
    pct_modified = modified / max(1, len(df)) * 100
    
    return {
        "rows_modified": int(modified),
        "pct_modified": round(float(pct_modified), 3),
        "max_correction_mw": float(diff.max()),
        "mean_correction_mw": float(diff.mean()),
    }
