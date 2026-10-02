import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.multi_horizon import HorizonModel, HORIZONS, INTRADAY_FEATURES, DAY_AHEAD_FEATURES


def test_day_ahead_horizons_exclude_production_lags():
    """J+1/J+2/J+3 must never use production-lag features (would not be
    operationally available that far ahead) - CDC continuation Phase 6.
    (Checks for the '_lag_'/'rollmean_'/'rollstd_' naming convention used by
    src/features/lag_features.py, not a bare 'lag' substring - which would
    also match unrelated names like 'daylight_flag'.)"""
    for h in ["J+1", "J+2", "J+3"]:
        feats = HORIZONS[h]["features"]
        assert not any("_lag_" in f or "rollmean_" in f or "rollstd_" in f for f in feats), \
            f"{h} features must not include production lags: {feats}"


def test_intraday_may_use_production_lags():
    assert any("lag" in f for f in INTRADAY_FEATURES)


def test_horizon_label_construction_is_strictly_future():
    n = 300
    df = pd.DataFrame({
        "district_id": ["D01"] * n,
        "timestamp": pd.date_range("2026-07-01", periods=n, freq="15min"),
        "y": range(n),
    })
    hm = HorizonModel(horizon_name="J+1", feature_cols=[])
    out = hm.build_supervised_frame(df, "y")
    # target_at_horizon at row i must equal y at i + steps, never <= i
    steps = HORIZONS["J+1"]["steps"]
    for i in range(0, n - steps, 50):
        assert out["target_at_horizon"].iloc[i] == df["y"].iloc[i + steps]
    assert (out["target_timestamp"] > out["forecast_issue_time"]).all()
