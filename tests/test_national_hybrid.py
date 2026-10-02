import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecasting import national_hybrid as NH  # noqa: E402


def test_hybrid_physical_bounds():
    d = pd.read_csv("data/processed/training_dataset_15min_sample.csv", parse_dates=["timestamp"])
    p = NH.hybrid_forecast_mw(d)
    assert (p >= 0).all() and (p <= d["capacity_mw"] + 1e-9).all()
    assert (p[d["solar_elevation_deg"] <= 0] == 0).all()


def test_transferred_interval_ordered():
    s = pd.Series([0.0, 1.0, 2.0])
    iv = NH.add_interval(s, pd.Series([5.0, 5.0, 5.0]), "J+1")
    assert (iv.p10_mw <= iv.p50_mw).all() and (iv.p50_mw <= iv.p90_mw).all()
