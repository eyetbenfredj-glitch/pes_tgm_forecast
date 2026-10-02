"""
Drift & anomaly monitoring (CDC continuation Phase 12).

Implements, with real (not fabricated) statistical methods:
  - weather distribution drift: Population Stability Index (PSI) between a
    reference window and a current window, per weather variable.
  - production distribution drift: same PSI method on the target series.
  - forecast-error drift: rolling MAE / rolling bias over time, flagged when
    a window's rolling MAE exceeds a multiple of the reference-period MAE.
  - capacity drift: simple check for month-over-month capacity jumps beyond
    what the 3 official snapshots justify (sanity check on the interpolation).

No ML/black box here - drift indicators are all classical, auditable statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1D samples. Standard bands:
    <0.1 no significant shift, 0.1-0.25 moderate shift, >0.25 major shift.
    Falls back to a simple standardized-mean-shift proxy when the reference
    sample has (near-)zero variance, since quantile-based binning degenerates
    to a single bin in that case and would otherwise silently report 0."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 10 or len(current) < 10:
        return float("nan")
    if np.std(reference) < 1e-9:
        # degenerate reference distribution: PSI's quantile binning cannot
        # resolve a shift, so measure it directly against a small tolerance.
        return float(min(10.0, abs(np.mean(current) - np.mean(reference)) / max(np.std(current), 1e-6)))
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_hist / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(len(current), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_band(value: float) -> str:
    if np.isnan(value):
        return "INSUFFICIENT_DATA"
    if value < 0.1:
        return "STABLE"
    if value < 0.25:
        return "MODERATE_SHIFT"
    return "MAJOR_SHIFT"


def weather_drift_report(df: pd.DataFrame, reference_end: pd.Timestamp,
                          variables: list[str] | None = None) -> pd.DataFrame:
    variables = variables or [
        "temperature_2m", "shortwave_radiation", "cloud_cover", "relative_humidity_2m", "wind_speed_10m",
    ]
    ref = df[df["timestamp"] <= reference_end]
    cur = df[df["timestamp"] > reference_end]
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        score = psi(ref[v].to_numpy(dtype=float), cur[v].to_numpy(dtype=float))
        rows.append({"variable": v, "psi": score, "status": psi_band(score)})
    return pd.DataFrame(rows)


def production_drift_report(df: pd.DataFrame, target_col: str, reference_end: pd.Timestamp) -> dict:
    ref = df[df["timestamp"] <= reference_end][target_col].to_numpy(dtype=float)
    cur = df[df["timestamp"] > reference_end][target_col].to_numpy(dtype=float)
    score = psi(ref, cur)
    return {"variable": target_col, "psi": score, "status": psi_band(score)}


def capacity_drift_report(cap_ts: pd.DataFrame, max_monthly_growth_pct: float = 50.0) -> pd.DataFrame:
    """Flags any district whose capacity jumps more than max_monthly_growth_pct
    in a 30-day window - a sanity check on the snapshot interpolation, not a
    claim about real fleet behaviour."""
    rows = []
    for did, g in cap_ts.sort_values("timestamp").groupby("district_id"):
        g = g.set_index("timestamp")
        growth = 100 * g["capacity_mw"].pct_change(freq="30D")
        flagged = growth[growth.abs() > max_monthly_growth_pct]
        for ts, val in flagged.items():
            rows.append({"district_id": did, "timestamp": ts, "growth_pct_30d": val})
    return pd.DataFrame(rows)


@dataclass
class ForecastErrorDrift:
    reference_window: int = 96 * 7   # 1 week of 15-min steps
    current_window: int = 96 * 7
    degradation_multiple: float = 1.5  # flag if current rolling MAE > 1.5x reference

    def report(self, df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
        df = df.sort_values("timestamp").copy()
        df["abs_error"] = (df[y_true_col] - df[y_pred_col]).abs()
        df["rolling_mae"] = df["abs_error"].rolling(self.current_window, min_periods=10).mean()
        df["rolling_bias"] = (df[y_pred_col] - df[y_true_col]).rolling(self.current_window, min_periods=10).mean()
        ref_mae = df["abs_error"].iloc[:self.reference_window].mean()
        df["reference_mae"] = ref_mae
        df["degraded"] = df["rolling_mae"] > self.degradation_multiple * ref_mae
        return df[["timestamp", "rolling_mae", "rolling_bias", "reference_mae", "degraded"]]


def anomaly_flags(df: pd.DataFrame, production_col: str, capacity_col: str = "capacity_mw",
                   daylight_col: str = "daylight_flag") -> pd.DataFrame:
    """Wraps src.validation.physical_checks for a monitoring-oriented flat anomaly table."""
    from src.validation.physical_checks import run_all_checks
    checked = run_all_checks(df, production_col=production_col, capacity_col=capacity_col, daylight_col=daylight_col)
    cols = ["timestamp", "district_id"] if "district_id" in df.columns else ["timestamp"]
    cols += [production_col, "physical_anomaly_flag"]
    return checked[[c for c in cols if c in checked.columns]]
