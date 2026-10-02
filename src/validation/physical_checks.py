"""
Physical sanity checks applied to any production series (real, proxy, or
forecast) - CDC section 18. Pure functions, no I/O, fully unit-testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_bounds(production_mw: pd.Series, capacity_mw: pd.Series, tol: float = 1e-6) -> pd.Series:
    """True where 0 <= production <= capacity (+tol)."""
    return (production_mw >= -tol) & (production_mw <= capacity_mw + tol)


def check_night_zero(production_mw: pd.Series, daylight_flag: pd.Series, tol: float = 1e-6) -> pd.Series:
    """True where night rows (daylight_flag == 0) have production ~ 0."""
    return np.where(daylight_flag == 0, production_mw.abs() <= tol, True)


def check_no_impossible_jumps(production_mw: pd.Series, capacity_mw: pd.Series,
                               max_ramp_fraction_per_step: float = 0.6) -> pd.Series:
    """
    Flags timestamps where |delta production| exceeds a fraction of installed
    capacity in a single step (a proxy for "impossible sudden jump"; the
    fraction is generous because 15-min ramps under fast-moving clouds are
    legitimately steep - this is a coarse anomaly flag, not a hard physical law).
    """
    if len(production_mw) == 0:
        return pd.Series([], dtype=bool, index=production_mw.index)
    delta = production_mw.diff().abs()
    threshold = max_ramp_fraction_per_step * capacity_mw
    ok = delta <= threshold
    ok.iloc[0] = True  # no previous value to compare
    return ok


def check_daylight_consistency(production_mw: pd.Series, solar_elevation_deg: pd.Series,
                                min_elevation_for_production: float = 0.0, tol: float = 1e-6) -> pd.Series:
    """True unless there's material production while the sun is below the horizon."""
    return ~((solar_elevation_deg <= min_elevation_for_production) & (production_mw.abs() > tol))


def run_all_checks(df: pd.DataFrame, production_col: str,
                    capacity_col: str = "capacity_mw",
                    daylight_col: str = "daylight_flag",
                    elevation_col: str = "solar_elevation_deg") -> pd.DataFrame:
    """Returns df with boolean check_* columns added and a summary attached via .attrs."""
    out = df.copy()
    out["check_bounds_ok"] = check_bounds(out[production_col], out[capacity_col])
    out["check_night_zero_ok"] = check_night_zero(out[production_col], out[daylight_col])
    out["check_no_jump_ok"] = check_no_impossible_jumps(out[production_col], out[capacity_col])
    if elevation_col in out.columns:
        out["check_daylight_consistency_ok"] = check_daylight_consistency(out[production_col], out[elevation_col])
    check_cols = [c for c in out.columns if c.startswith("check_") and c.endswith("_ok")]
    out["physical_anomaly_flag"] = ~out[check_cols].all(axis=1)
    summary = {c: float(out[c].mean()) for c in check_cols}
    summary["pct_flagged_anomalous"] = float(out["physical_anomaly_flag"].mean() * 100)
    out.attrs["physical_check_summary"] = summary
    return out


if __name__ == "__main__":
    df = pd.DataFrame({
        "capacity_mw": [10, 10, 10, 10],
        "pv_production_mw_proxy": [0, 5, 11, 0],   # row index 2 violates bounds (>capacity)
        "daylight_flag": [0, 1, 1, 0],
        "solar_elevation_deg": [-5, 20, 40, -2],
    })
    out = run_all_checks(df, production_col="pv_production_mw_proxy")
    print(out)
    print(out.attrs["physical_check_summary"])
