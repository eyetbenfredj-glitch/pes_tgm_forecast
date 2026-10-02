"""Tests for the REAL-data (MEASURED_REAL ENSTAB) forecasting pipeline."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecasting import real_pv as R  # noqa: E402

CAL = json.loads(Path("configs/calibration_enstab.json").read_text())


@pytest.fixture(scope="module")
def df():
    full = R.load_enstab()
    return R.add_targets(full.iloc[: 96 * 60], CAL["c_pcs"])      # first 60 days: fast


def test_clearsky_zero_at_night_and_daytime_flag(df):
    assert (df.loc[df["elev"] < -5, "pcs"] == 0).all()
    assert df["day"].sum() > 1000


def test_apply_constraints_physical():
    pcs = np.array([0.0, 5.0, 500.0, 500.0])
    out = R.apply_constraints(np.array([900.0, 900.0, 99999.0, -50.0]), pcs)
    assert out[0] == 0 and out[1] == 0           # night -> 0
    assert out[2] == R.PNOM_W and out[3] == 0    # 0 <= P <= Pnom


@pytest.mark.parametrize("hz", ["15min", "3h", "J+1", "J+3"])
def test_no_future_leakage(df, hz):
    """Corrupt EVERYTHING strictly after a given issue time: features of that issue must not change."""
    feats = R.base_features(df)
    X = R.build_dataset(df, feats, hz)
    issue = pd.Timestamp("2022-03-30 10:00")
    before = X[X["issue"] == issue].drop(columns=["p", "y", "p_persist"]).reset_index(drop=True)
    assert len(before) > 0
    bad = df.copy()
    future = bad.index > issue
    for c in ["p", "ghi", "t_amb", "t_cell", "rh", "wind", "pres"]:
        bad.loc[future, c] = 1e6
    bad = R.add_targets(bad, CAL["c_pcs"])
    X2 = R.build_dataset(bad, R.base_features(bad), hz)
    after = X2[X2["issue"] == issue].drop(columns=["p", "y", "p_persist"]).reset_index(drop=True)
    feature_cols = [c for c in R.FEATURE_SETS["full"]]
    pd.testing.assert_frame_equal(before[feature_cols], after[feature_cols], check_exact=False, rtol=1e-9)


def test_splits_are_chronological_and_disjoint():
    ends = [pd.Timestamp(v[1]) for v in R.SPLITS.values()]
    starts = [pd.Timestamp(v[0]) for v in R.SPLITS.values()]
    assert starts[1] > ends[0] and starts[2] > ends[1]


def test_conformal_restores_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)       # deliberately too narrow
    qhat = R.conformal_qhat(lo[:2500], hi[:2500], y[:2500], alpha=0.2)
    cov = np.mean((y[2500:] >= lo[2500:] - qhat) & (y[2500:] <= hi[2500:] + qhat))
    assert 0.77 <= cov <= 0.84


# ---- artifacts produced by scripts/12_real_benchmark.py
M = pd.read_csv("data/validation/real_benchmark_metrics.csv")
C = pd.read_csv("data/validation/real_benchmark_coverage.csv")


def _n(model, hz):
    return M[(M.model == model) & (M.horizon == hz)].nMAE_pct.iloc[0]


@pytest.mark.parametrize("hz", ["15min", "1h", "3h", "6h", "J+1"])
def test_ml_beats_persistence(hz):
    assert _n("LightGBM", hz) < _n("Persistence", hz)


def test_conformal_coverage_close_to_nominal():
    c = C[C.interval == "conformal"].empirical_pct
    assert c.between(72, 90).all(), c.tolist()


def test_calibrated_physics_unbiased():
    p = pd.read_csv("data/validation/real_physics_transfer.csv").set_index("model")
    cal_row = [i for i in p.index if i.startswith("Calibrated")][0]
    assert abs(p.loc[cal_row, "bias_W"]) < 50
    assert p.loc[cal_row, "MAE_W"] < p.loc[[i for i in p.index if i.startswith("National")][0], "MAE_W"]


def test_forecast_file_physical():
    f = pd.read_csv("data/validation/real_test_forecasts.csv.gz")
    assert (f[["lgbm_w", "p10_w", "p90_w"]] >= 0).all().all()
    assert (f.lgbm_w <= R.PNOM_W).all()
    assert (f.p10_w <= f.p90_w + 1e-6).all()
    night = f.clearsky_w < R.DAY_PCS_MIN_W - 0.5   # csv stores 1 decimal
    assert (f.loc[night, "lgbm_w"] == 0).all()
