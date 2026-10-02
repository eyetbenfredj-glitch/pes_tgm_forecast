"""
National hybrid forecast: physics calibrated on MEASURED ENSTAB data, applied to the fleet.

    P_district(t) = capacity_mw * A_norm * G(t) * (1 + gamma * (Tcell(t) - 25)),   Tcell = Tamb + (NOCT-20)/800 * G

A_norm and gamma are fitted on the ENSTAB train split (scripts/12_real_benchmark.py ->
configs/calibration_enstab.json) instead of the arbitrary PR = 0.80 used by the demo target.
G is the weather irradiance in the dataset: SYNTHETIC_CLEARSKY in the shipped sample, real
Open-Meteo forecast once scripts/02 is run with network access.

Uncertainty is TRANSFERRED from the ENSTAB J+1 conformal intervals (ratio actual/forecast).
It is single-site and therefore conservative for aggregated levels - labelled as such.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

CAL_PATH = Path("configs/calibration_enstab.json")
FC_PATH = Path("data/validation/real_test_forecasts.csv.gz")
NOCT = 45.0
FCOL = "pv_forecast_mw_hybrid"


@lru_cache(maxsize=1)
def load_calibration() -> dict:
    return json.loads(CAL_PATH.read_text())["physics_calibration"]


def hybrid_forecast_mw(df: pd.DataFrame) -> pd.Series:
    cal = load_calibration()
    g = df["shortwave_radiation"].astype(float).clip(lower=0)
    t_cell = df["temperature_2m"].astype(float) + (NOCT - 20.0) / 800.0 * g
    cf = cal["A_norm"] * g * (1 + cal["gamma_per_c"] * (t_cell - 25.0))
    p = df["capacity_mw"] * cf.clip(0, 1)
    if "solar_elevation_deg" in df.columns:
        p = p.where(df["solar_elevation_deg"] > 0, 0.0)
    return p.clip(lower=0).rename(FCOL)


@lru_cache(maxsize=4)
def transfer_ratios(horizon: str = "J+1", lo_q: float = 0.10, hi_q: float = 0.90) -> tuple[float, float, str]:
    """Quantiles of actual/forecast on ENSTAB test rows (forecast > 300 W). Returns (r_lo, r_hi, note)."""
    f = pd.read_csv(FC_PATH, usecols=["horizon", "actual_w", "lgbm_w"])
    f = f[(f.horizon == horizon) & (f.lgbm_w > 300)]
    r = (f.actual_w / f.lgbm_w).clip(0, 3)
    return float(r.quantile(lo_q)), float(r.quantile(hi_q)), (
        f"interval transferred from ENSTAB {horizon} (single site, n={len(f)}); conservative for aggregates")


def add_interval(forecast: pd.Series, capacity: pd.Series, horizon: str = "J+1") -> pd.DataFrame:
    lo, hi, note = transfer_ratios(horizon)
    out = pd.DataFrame({"p50_mw": forecast})
    out["p10_mw"] = (forecast * lo).clip(lower=0)
    out["p90_mw"] = np.minimum(forecast * hi, capacity)
    out["interval_note"] = note
    return out
