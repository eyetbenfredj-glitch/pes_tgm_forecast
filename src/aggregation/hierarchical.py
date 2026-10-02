"""
Hierarchical forecasting reconciliation (CDC section 13/14).

Method chosen: BOTTOM-UP. Rationale (documented, per CDC requirement to
explain the choice): district-level forecasts are the model's native output
(one row per district, per timestamp); governorate and national levels are
pure sums of their constituent districts. Bottom-up is:
  - trivially mathematically coherent (sum(children) == parent, exactly, by
    construction - no reconciliation residual to estimate),
  - requires no additional model or estimated reconciliation matrix (unlike
    MinT, which needs a forecast-error covariance estimate that is not
    meaningful yet given no real production data to estimate it from),
  - appropriate because the CDC's forecasting unit is inherently the
    district (that's where fleet capacity and weather are grounded).
A top-down or MinT alternative is a reasonable future upgrade once real
district-level production data lets you estimate a genuine error-covariance
structure - documented here rather than implemented against synthetic data
that would just recover the identity mapping.
"""
from __future__ import annotations

import pandas as pd


def bottom_up_governorate(df: pd.DataFrame, value_col: str,
                           group_cols: list[str] = ("timestamp", "governorate")) -> pd.DataFrame:
    return df.groupby(list(group_cols), as_index=False)[value_col].sum()


def bottom_up_national(df: pd.DataFrame, value_col: str,
                        group_cols: list[str] = ("timestamp",)) -> pd.DataFrame:
    out = df.groupby(list(group_cols), as_index=False)[value_col].sum()
    out["spatial_level"] = "national"
    return out


def reconcile_all_levels(district_df: pd.DataFrame, value_col: str) -> dict[str, pd.DataFrame]:
    """Returns {'district':..., 'governorate':..., 'national':...} all built
    from the SAME district-level values, so consistency is exact by
    construction (verified in tests/test_reconciliation.py, not just assumed)."""
    district = district_df[["timestamp", "district_id", "district", "governorate", value_col]].copy()
    governorate = bottom_up_governorate(district, value_col)
    national = bottom_up_national(district, value_col)
    return {"district": district, "governorate": governorate, "national": national}


def check_consistency(levels: dict[str, pd.DataFrame], value_col: str, tol: float = 1e-6) -> dict:
    """Verifies sum(district) == governorate and sum(governorate) == national,
    per timestamp. Returns a summary dict; raises no exception (callers decide
    what to do with a failure) so it can double as a monitoring probe."""
    district, governorate, national = levels["district"], levels["governorate"], levels["national"]

    d_to_g = district.groupby(["timestamp", "governorate"], as_index=False)[value_col].sum()
    check_g = d_to_g.merge(governorate, on=["timestamp", "governorate"], suffixes=("_sum_district", "_governorate"))
    check_g["diff"] = (check_g[f"{value_col}_sum_district"] - check_g[f"{value_col}_governorate"]).abs()
    governorate_consistent = bool((check_g["diff"] <= tol).all())

    g_to_n = governorate.groupby("timestamp", as_index=False)[value_col].sum()
    check_n = g_to_n.merge(national, on="timestamp", suffixes=("_sum_governorate", "_national"))
    check_n["diff"] = (check_n[f"{value_col}_sum_governorate"] - check_n[f"{value_col}_national"]).abs()
    national_consistent = bool((check_n["diff"] <= tol).all())

    return {
        "governorate_consistent": governorate_consistent,
        "national_consistent": national_consistent,
        "max_governorate_diff": float(check_g["diff"].max()) if len(check_g) else 0.0,
        "max_national_diff": float(check_n["diff"].max()) if len(check_n) else 0.0,
    }


if __name__ == "__main__":
    df = pd.DataFrame({
        "timestamp": ["2026-07-01T12:00"] * 4,
        "district_id": ["D01", "D02", "D03", "D04"],
        "district": ["A", "B", "C", "D"],
        "governorate": ["G1", "G1", "G2", "G2"],
        "forecast_mw": [10.0, 5.0, 3.0, 7.0],
    })
    levels = reconcile_all_levels(df, "forecast_mw")
    for k, v in levels.items():
        print(f"--- {k} ---")
        print(v)
    print(check_consistency(levels, "forecast_mw"))
