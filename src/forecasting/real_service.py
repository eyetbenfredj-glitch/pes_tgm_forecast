"""
Live inference over the trained REAL-data models (models/registry/real_enstab/*.joblib).

`predict(issue_time, horizon)` rebuilds the leak-free feature vector from the measured history up
to `issue_time` and runs the saved LightGBM mean + quantile models + conformal correction.
It is a replay of a genuine operational forecast: nothing after `issue_time` is used as input.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import real_pv as R

MODEL_DIR = Path("models/registry/real_enstab")
CAL_PATH = Path("configs/calibration_enstab.json")
_LOCK = threading.Lock()
_STATE: dict = {}


def _boot():
    with _LOCK:
        if "df" in _STATE:
            return
        c = json.loads(CAL_PATH.read_text())["c_pcs"]
        df = R.add_targets(R.load_enstab(), c)
        _STATE.update(df=df, feats=R.base_features(df), ds={}, models={})


def _dataset(hz: str) -> pd.DataFrame:
    _boot()
    if hz not in _STATE["ds"]:
        _STATE["ds"][hz] = R.build_dataset(_STATE["df"], _STATE["feats"], hz)
    return _STATE["ds"][hz]


def _models(hz: str) -> dict:
    if hz not in _STATE.get("models", {}):
        _boot()
        _STATE["models"][hz] = joblib.load(MODEL_DIR / f"{hz}.joblib")
    return _STATE["models"][hz]


def available_issue_range() -> tuple[str, str]:
    _boot()
    idx = _STATE["df"].index
    return str(idx[2 * R.STEPS_PER_DAY]), str(idx[-1])


def predict(issue_time: str, horizon: str) -> list[dict]:
    if horizon not in R.HORIZONS:
        raise ValueError(f"horizon must be one of {list(R.HORIZONS)}")
    X = _dataset(horizon)
    ts = pd.Timestamp(issue_time)
    if R.HORIZONS[horizon][0] == "dayahead":
        ts = ts.normalize() + pd.Timedelta(hours=R.ISSUE_HOUR)      # snap to the 10:00 gate
    else:
        ts = ts.floor("15min")
    sub = X[X["issue"] == ts]
    if sub.empty:
        raise LookupError(f"no forecast available for issue time {ts} (outside data range?)")
    m = _models(horizon)
    F = m["features"]
    pcs = sub["pcs"].values
    p50 = R.apply_constraints(m["mean"].predict(sub[F]) * pcs, pcs)
    lo, hi = np.sort(np.c_[m["q10"].predict(sub[F]), m["q90"].predict(sub[F])], axis=1).T
    p10 = R.apply_constraints(np.maximum(lo - m["qhat"], 0) * pcs, pcs)
    p90 = R.apply_constraints((hi + m["qhat"]) * pcs, pcs)
    return [dict(issue_time=str(ts), target_time=str(t), horizon=horizon, forecast_w=float(a),
                 p10_w=float(b), p90_w=float(c), clearsky_w=float(d),
                 measured_w=float(e) if np.isfinite(e) else None,
                 data_source="MEASURED_REAL_ENSTAB (replay, no future data used)")
            for t, a, b, c, d, e in zip(sub["t"], p50, p10, p90, pcs, sub["p"])]
