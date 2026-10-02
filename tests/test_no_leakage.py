import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.lag_features import add_lag_features, add_rolling_features
from src.forecasting.backtesting import shift_to_horizon


def _toy_df(n=20):
    return pd.DataFrame({
        "district_id": ["D01"] * n,
        "timestamp": pd.date_range("2026-07-01", periods=n, freq="15min"),
        "y": np.arange(n, dtype=float),
    })


def test_lag_feature_never_equals_future_value():
    df = _toy_df()
    out = add_lag_features(df, "y", lag_steps={"lag_1": 1})
    # lag_1 at row i must equal y at row i-1, and must NEVER equal y at row i or later
    for i in range(1, len(out)):
        assert out["y_lag_1"].iloc[i] == out["y"].iloc[i - 1]
        assert out["y_lag_1"].iloc[i] != out["y"].iloc[i]


def test_lag_feature_row0_is_nan():
    df = _toy_df()
    out = add_lag_features(df, "y", lag_steps={"lag_1": 1})
    assert pd.isna(out["y_lag_1"].iloc[0])


def test_rolling_feature_excludes_current_row():
    df = _toy_df()
    out = add_rolling_features(df, "y", windows={"3": 3})
    # rollmean_3 at row i must be the mean of y[i-3:i] (exclusive of i), i.e.
    # NOT include the current row's own value.
    row = 10
    expected = df["y"].iloc[row - 3:row].mean()
    assert abs(out["y_rollmean_3"].iloc[row] - expected) < 1e-9


def test_horizon_label_is_strictly_future():
    df = _toy_df()
    label = shift_to_horizon(df, "y", horizon_steps=4)
    for i in range(len(df) - 4):
        assert label.iloc[i] == df["y"].iloc[i + 4]
    # last 4 rows have no future value -> NaN, never back-filled from the past
    assert label.iloc[-1:].isna().all()
