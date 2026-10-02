import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring.drift import psi, psi_band, weather_drift_report, ForecastErrorDrift


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    score = psi(x, x.copy())
    assert score < 0.01
    assert psi_band(score) == "STABLE"


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(5, 1, 1000)  # large shift
    score = psi(ref, cur)
    assert score > 0.25
    assert psi_band(score) == "MAJOR_SHIFT"


def test_weather_drift_report_runs():
    n = 400
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "temperature_2m": np.concatenate([
            rng.normal(15.0, 1.5, 200),   # reference period: winter-like
            rng.normal(30.0, 1.5, 200),   # current period: large real shift
        ]),
    })
    report = weather_drift_report(df, reference_end=df["timestamp"].iloc[199], variables=["temperature_2m"])
    assert report.iloc[0]["status"] == "MAJOR_SHIFT"


def test_forecast_error_drift_flags_degradation():
    n = 500
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    y_true = np.full(n, 10.0)
    y_pred = np.where(np.arange(n) < 250, 10.0, 10.0 + np.r_[np.zeros(250), np.linspace(0, 5, 250)])
    df = pd.DataFrame({"timestamp": ts, "y_true": y_true, "y_pred": y_pred})
    fed = ForecastErrorDrift(reference_window=100, current_window=50, degradation_multiple=1.5)
    report = fed.report(df, "y_true", "y_pred")
    assert report["degraded"].iloc[-1]  # error grew a lot by the end
    assert not report["degraded"].iloc[150]  # stable region not flagged
