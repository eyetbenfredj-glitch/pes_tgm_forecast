"""
Lag / rolling-window features for a per-district production or weather
series (CDC section 9, FEATURE GROUP 5).

LEAKAGE SAFETY: every feature here is computed using pandas .shift()
BEFORE any rolling statistic, so a row's features only ever look backward
in time relative to that row. No feature is ever computed using rows at
or after its own timestamp. This is unit-tested in tests/test_no_leakage.py.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_LAG_STEPS_15MIN = {
    "lag_1": 1,          # 15 min ago
    "lag_4": 4,           # 1 hour ago
    "lag_96": 96,          # same time yesterday (15-min steps: 96/day)
    "lag_672": 672,         # same time last week
}


def add_lag_features(df: pd.DataFrame, value_col: str, group_col: str = "district_id",
                      lag_steps: dict | None = None) -> pd.DataFrame:
    lag_steps = lag_steps or DEFAULT_LAG_STEPS_15MIN
    df = df.sort_values([group_col, "timestamp"]).copy()
    for name, steps in lag_steps.items():
        df[f"{value_col}_{name}"] = df.groupby(group_col)[value_col].shift(steps)
    return df


def add_rolling_features(df: pd.DataFrame, value_col: str, group_col: str = "district_id",
                          windows: dict | None = None) -> pd.DataFrame:
    """
    Rolling mean/std computed on the ALREADY-SHIFTED lag_1 series (not on the
    raw series), so the rolling window for row t only ever includes t-1 and
    earlier - this is what prevents leakage from a rolling window that would
    otherwise include the current (unobserved-at-forecast-time) value.
    """
    windows = windows or {"4": 4, "96": 96}  # 1h, 1 day (in 15-min steps)
    df = df.sort_values([group_col, "timestamp"]).copy()
    base = df.groupby(group_col)[value_col].shift(1)
    for name, w in windows.items():
        df[f"{value_col}_rollmean_{name}"] = base.groupby(df[group_col]).transform(
            lambda s: s.rolling(int(w), min_periods=max(1, int(w) // 2)).mean()
        )
        df[f"{value_col}_rollstd_{name}"] = base.groupby(df[group_col]).transform(
            lambda s: s.rolling(int(w), min_periods=max(1, int(w) // 2)).std()
        )
    return df


if __name__ == "__main__":
    import numpy as np
    n = 10
    df = pd.DataFrame({
        "district_id": ["D01"] * n,
        "timestamp": pd.date_range("2026-07-01", periods=n, freq="15min"),
        "pv_production_mw_proxy": np.arange(n, dtype=float),
    })
    out = add_lag_features(df, "pv_production_mw_proxy", lag_steps={"lag_1": 1, "lag_2": 2})
    out = add_rolling_features(out, "pv_production_mw_proxy", windows={"3": 3})
    print(out[["timestamp", "pv_production_mw_proxy", "pv_production_mw_proxy_lag_1",
               "pv_production_mw_proxy_lag_2", "pv_production_mw_proxy_rollmean_3"]])
