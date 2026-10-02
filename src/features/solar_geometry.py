"""
Solar geometry features (CDC section 8, FEATURE GROUP 3), computed with pvlib
(an established, peer-reviewed solar-position library) - not invented formulas.

Provides: solar elevation, solar zenith, solar azimuth, daylight flag,
clear-sky GHI/DNI/DHI (Ineichen-Perez model with climatological turbidity),
sunrise/sunset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def add_solar_geometry(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    ts_col: str = "timestamp",
    altitude_m: float = 50.0,
) -> pd.DataFrame:
    """
    df must contain one row per (location, timestamp). Because pvlib's solar
    position call is vectorized over time for a FIXED location, this groups
    by (lat, lon) to stay efficient for a 50-district panel.
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])

    results = []
    for (lat, lon), g in df.groupby([lat_col, lon_col]):
        times = pd.DatetimeIndex(g[ts_col]).tz_localize("Africa/Tunis", ambiguous="NaT", nonexistent="NaT")
        valid = ~times.isna()
        solpos = pvlib.solarposition.get_solarposition(times[valid], lat, lon, altitude=altitude_m)
        linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times[valid], lat, lon)
        airmass_rel = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])
        pressure = pvlib.atmosphere.alt2pres(altitude_m)
        airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel, pressure)
        cs = pvlib.clearsky.ineichen(
            solpos["apparent_zenith"], airmass_abs, linke_turbidity, altitude=altitude_m
        )

        g2 = g.loc[valid].copy()
        g2["solar_zenith_deg"] = solpos["apparent_zenith"].values
        g2["solar_elevation_deg"] = solpos["apparent_elevation"].values
        g2["solar_azimuth_deg"] = solpos["azimuth"].values
        g2["daylight_flag"] = (g2["solar_elevation_deg"] > 0).astype(int)
        g2["clearsky_ghi_wm2"] = cs["ghi"].values
        g2["clearsky_dni_wm2"] = cs["dni"].values
        g2["clearsky_dhi_wm2"] = cs["dhi"].values
        results.append(g2)

    out = pd.concat(results, ignore_index=True).sort_values([lat_col, lon_col, ts_col])

    # Cyclic temporal encodings (CDC FEATURE GROUP 4)
    ts = pd.to_datetime(out[ts_col])
    hour_frac = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
    doy = ts.dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["day_of_week"] = ts.dt.dayofweek
    out["month"] = ts.dt.month

    return out.reset_index(drop=True)


if __name__ == "__main__":
    sample = pd.DataFrame({
        "latitude": [34.7406] * 96,
        "longitude": [10.7603] * 96,
        "timestamp": pd.date_range("2026-07-01", periods=96, freq="15min"),
    })
    out = add_solar_geometry(sample)
    print(out[["timestamp", "solar_elevation_deg", "clearsky_ghi_wm2", "daylight_flag"]].iloc[28:40])
