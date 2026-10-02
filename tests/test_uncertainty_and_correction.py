import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.uncertainty.quantile_forecast import QuantileForecastModel, QUANTILE_COL_NAMES, evaluate_coverage
from src.calibration.residual_correction import RollingBiasCorrector


def _toy_regression_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1000, n)
    y = 0.01 * x + rng.normal(0, 2, n)
    return pd.DataFrame({
        "solar_elevation_deg": x / 1000 * 90,
        "cloud_cover": rng.uniform(0, 100, n),
        "capacity_mw": 10.0,
        "shortwave_radiation": x,
        "target": np.clip(y, 0, None),
    })


def test_quantile_ordering_respected():
    df = _toy_regression_df()
    train, calib, test = df.iloc[:200], df.iloc[200:300], df.iloc[300:]
    m = QuantileForecastModel(feature_cols=["shortwave_radiation", "cloud_cover"])
    m.fit(train, "target")
    m.calibrate(calib, "target")
    preds = m.predict(test)
    cols = [QUANTILE_COL_NAMES[t] for t in [0.10, 0.25, 0.50, 0.75, 0.90]]
    arr = preds[cols].to_numpy()
    assert (np.diff(arr, axis=1) >= -1e-9).all()


def test_rolling_bias_corrector_reduces_systematic_bias():
    n = 200
    rng = np.random.default_rng(1)
    ts = pd.date_range("2026-07-01", periods=n, freq="15min")
    actual = 5.0 + rng.normal(0, 0.1, n)
    forecast = actual - 1.0  # systematic underforecast bias of 1 MW
    df = pd.DataFrame({
        "district_id": ["D01"] * n, "timestamp": ts,
        "target": actual, "raw_forecast_mw": forecast, "capacity_mw": 10.0,
    })
    hist, ev = df.iloc[:100], df.iloc[100:]
    corrector = RollingBiasCorrector(window=1000)
    corrector.fit(hist, actual_col="target", forecast_col="raw_forecast_mw")
    corrected = corrector.correct(ev, forecast_col="raw_forecast_mw")

    raw_mae = (ev["target"] - ev["raw_forecast_mw"]).abs().mean()
    corrected_mae = (ev["target"] - corrected).abs().mean()
    assert corrected_mae < raw_mae  # correction must actually reduce error
