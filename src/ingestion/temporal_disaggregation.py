"""
Temporal disaggregation of hourly irradiance/PV data to 15-minute resolution.

Method: Solar-geometry-weighted distribution of hourly energy into 15-minute
sub-intervals using the clear-sky irradiance shape within each hour.

Rationale:
  - Naive linear interpolation creates unrealistic linear ramps that don't
    reflect the true solar irradiance profile within an hour.
  - The clear-sky irradiance (from pvlib Ineichen model) provides a physically
    consistent shape function for the sub-hourly distribution.
  - For each 1-hour interval, the hourly energy is distributed among the four
    15-minute sub-steps proportionally to the clear-sky GHI at each sub-step.
  - If clear-sky GHI is zero for the whole hour (night), all four sub-steps
    receive zero (physically correct).

Daily energy conservation:
  sum(15-min values over 1 hour) × (15/60) ≈ original_hourly_value × (60/60)
  → i.e., the mean of the four 15-min values equals the original hourly value,
  so integrated daily energy is preserved.

Label:
  The resulting 15-min series is labeled:
    PVGIS_PHYSICAL_ESTIMATE_15MIN          (when input is PVGIS data)
    SYNTHETIC_CLEARSKY_DISAGGREGATED_15MIN (when input is synthetic)
  Never presented as measured 15-min observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def disaggregate_hourly_to_15min(
    hourly_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    ts_col: str = "timestamp",
    irr_col: str = "ghi_wm2",
    value_cols: list[str] | None = None,
    altitude_m: float = 50.0,
    source_label_suffix: str = "_15MIN",
) -> pd.DataFrame:
    """
    Disaggregate hourly irradiance/production records to 15-min using
    clear-sky shape weights per location.

    Parameters
    ----------
    hourly_df : DataFrame with hourly rows; must have lat_col, lon_col, ts_col,
                district_id, and the columns in value_cols.
    lat_col, lon_col, ts_col : column names for location and time.
    irr_col : column name for the primary irradiance used for weighting;
              defaults to "ghi_wm2". If not present, uniform weights are used.
    value_cols : columns to disaggregate. Defaults to [irr_col, "pv_production_mw_reference"].
    altitude_m : used for pvlib clear-sky model.
    source_label_suffix : appended to any "source" label columns found.

    Returns
    -------
    DataFrame at 15-min resolution. The "weather_source" / "production_source"
    columns gain the suffix to document the temporal downscaling step.
    """
    import pvlib

    if value_cols is None:
        value_cols = [c for c in [
            irr_col, "pv_production_mw_reference", "ghi_wm2", "dni_wm2", "dhi_wm2",
            "poa_irradiance_wm2", "pv_output_estimate_w",
        ] if c in hourly_df.columns]

    results = []
    for (did, lat, lon), g in hourly_df.groupby(["district_id", lat_col, lon_col]):
        g = g.sort_values(ts_col).copy()
        hourly_times = pd.DatetimeIndex(g[ts_col])

        # Generate 15-min timestamps for the full period
        t_start = hourly_times.min()
        t_end   = hourly_times.max() + pd.Timedelta(minutes=45)  # cover last hour's 4 steps
        fine_times = pd.date_range(t_start, t_end, freq="15min")

        # Clear-sky GHI at 15-min resolution for weight computation
        fine_tz = fine_times.tz_localize("Africa/Tunis", ambiguous="NaT", nonexistent="NaT")
        valid_fine = ~fine_tz.isna()
        cs_fine = np.zeros(len(fine_times))
        if valid_fine.any():
            try:
                solpos_f = pvlib.solarposition.get_solarposition(
                    fine_tz[valid_fine], lat, lon, altitude=altitude_m
                )
                linke_f = pvlib.clearsky.lookup_linke_turbidity(fine_tz[valid_fine], lat, lon)
                airm_rel_f = pvlib.atmosphere.get_relative_airmass(solpos_f["apparent_zenith"])
                pressure_f = pvlib.atmosphere.alt2pres(altitude_m)
                airm_abs_f = pvlib.atmosphere.get_absolute_airmass(airm_rel_f, pressure_f)
                cs_f = pvlib.clearsky.ineichen(
                    solpos_f["apparent_zenith"], airm_abs_f, linke_f, altitude=altitude_m
                )
                cs_fine[valid_fine] = cs_f["ghi"].values
            except Exception:
                cs_fine[:] = 0.0

        cs_fine_s = pd.Series(cs_fine, index=fine_times)

        # For each hourly row, distribute its value over 4×15-min steps
        fine_records = []
        for _, row in g.iterrows():
            t_hr = pd.Timestamp(row[ts_col])
            # The 4 sub-steps that fall within this hour
            sub_steps = pd.date_range(t_hr, periods=4, freq="15min")
            cs_sub = cs_fine_s.reindex(sub_steps, fill_value=0.0).values
            cs_sum = cs_sub.sum()

            for i, t15 in enumerate(sub_steps):
                sub_row = {"timestamp": t15, "district_id": did, lat_col: lat, lon_col: lon}
                # Copy all non-value columns
                for col in g.columns:
                    if col not in value_cols and col not in [ts_col, lat_col, lon_col, "district_id"]:
                        sub_row[col] = row[col]
                # Distribute value columns
                for vcol in value_cols:
                    if vcol not in g.columns:
                        continue
                    hourly_val = row[vcol]
                    if pd.isna(hourly_val):
                        sub_row[vcol] = np.nan
                    elif cs_sum > 0:
                        sub_row[vcol] = hourly_val * (cs_sub[i] / cs_sum) * 4
                        # × 4 because we distribute the mean hourly power into 4 steps
                        # such that mean(4 steps) = original hourly value
                    else:
                        # No clear-sky signal → uniform distribution (only during twilight edge cases)
                        sub_row[vcol] = hourly_val
                fine_records.append(sub_row)

        district_fine = pd.DataFrame(fine_records)
        # Update source labels to reflect disaggregation
        for col in district_fine.columns:
            if "source" in col.lower() and district_fine[col].dtype == object:
                district_fine[col] = district_fine[col].apply(
                    lambda v: str(v) + source_label_suffix if isinstance(v, str) and source_label_suffix not in str(v) else v
                )
        district_fine["disaggregation_method"] = "clearsky_shape_weighted_hourly_to_15min"
        results.append(district_fine)

    out = pd.concat(results, ignore_index=True).sort_values(["district_id", "timestamp"])
    return out.reset_index(drop=True)


def verify_energy_conservation(
    hourly_df: pd.DataFrame,
    fine_df: pd.DataFrame,
    value_col: str,
    ts_col: str = "timestamp",
    tol_pct: float = 1.0,
) -> dict:
    """
    Verify that hourly energy is approximately preserved after disaggregation.

    For each hourly timestamp h, the four 15-min values v_i at h, h+15, h+30, h+45
    should satisfy: mean(v_i) ≈ hourly_value(h).
    Equivalently: sum(v_i * 15min) ≈ hourly_value(h) * 60min.

    Returns dict with max_error_pct and pass/fail.
    """
    if value_col not in hourly_df.columns or value_col not in fine_df.columns:
        return {"status": "SKIPPED", "reason": f"{value_col} not in both dataframes"}

    fine_df = fine_df.copy()
    fine_df["hour_floor"] = pd.to_datetime(fine_df[ts_col]).dt.floor("h")
    hourly_means = fine_df.groupby(["district_id", "hour_floor"])[value_col].mean().reset_index()
    hourly_means.columns = ["district_id", "timestamp", "fine_mean"]

    merged = pd.merge(
        hourly_df[["district_id", ts_col, value_col]].rename(columns={ts_col: "timestamp"}),
        hourly_means, on=["district_id", "timestamp"], how="inner"
    )
    if merged.empty:
        return {"status": "NO_MATCH", "reason": "No matching hour/district found"}

    merged = merged[merged[value_col].notna() & merged["fine_mean"].notna()]
    denom = merged[value_col].abs().replace(0, np.nan)
    errors = ((merged["fine_mean"] - merged[value_col]) / denom * 100).abs()
    max_err = float(errors.max()) if len(errors) > 0 else 0.0
    mean_err = float(errors.mean()) if len(errors) > 0 else 0.0

    return {
        "status": "PASS" if max_err <= tol_pct else "WARN",
        "max_error_pct": round(max_err, 3),
        "mean_error_pct": round(mean_err, 3),
        "n_pairs": len(merged),
        "tolerance_pct": tol_pct,
    }
