"""
ENSTAB / Borj Cedria real PV production dataset validator and preparator.

DATA CLASSIFICATION: MEASURED_REAL
Source: ENSTAB (École Nationale des Sciences et Technologie Avancées de Borj Cedria),
        Ben Arous Governorate, Tunisia.
Location: ~36.72 °N, ~10.43 °E (Borj Cedria)
System: Small rooftop PV installation (~3 kWc), measured at 5-minute resolution.

This dataset is the ONLY confirmed MEASURED_REAL PV production data available
in this project. It MUST:
  - NEVER be merged into national training data.
  - NEVER be labeled as anything other than MEASURED_REAL_ENSTAB.
  - Be used ONLY as an external hold-out validation set.
  - Be compared against national model predictions WITHOUT the model having
    been trained on the ENSTAB test period.

Column mapping (raw → standard):
  Unnamed: 0        → timestamp       (index, datetime)
  Power             → power_w         (measured PV output, Watts)
  Irradiation       → ghi_wm2         (measured GHI, W/m²)
  Temperature       → temperature_2m_c (ambient temperature, °C)
  Cell_temperature  → cell_temperature_c (module cell temperature, °C)
  Humidity          → relative_humidity_pct (%)
  Wind_speed        → wind_speed_ms   (m/s)
  Pressure          → pressure_hpa    (hPa)

System characteristics (derived from data inspection):
  - Resolution: 5 minutes (uniform, no gaps)
  - Time range: 2022-02-24 → 2024-05-31
  - Max power: ~2,350 W  → estimated installed capacity ~3.0–3.5 kWc
  - No missing values, no negative power, no duplicate timestamps
  - Power units: WATTS (W), not kW or MW
  - Irradiation units: W/m²
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Provenance constants ──────────────────────────────────────────────────────
PRODUCTION_SOURCE = "MEASURED_REAL_ENSTAB"
LOCATION_SOURCE   = "ENSTAB_Borj_Cedria_Ben_Arous_Tunisia"

# Representative coordinates for Borj Cedria ENSTAB campus
ENSTAB_LATITUDE  = 36.720
ENSTAB_LONGITUDE = 10.430
ENSTAB_ALTITUDE_M = 30.0

# Nominal installed capacity: estimated from data (max observed power ÷ STC
# irradiance × derating factor). Do NOT claim this is a nameplate measurement.
# Derivation: max_power ≈ 2350 W at GHI ≈ 1000 W/m² → capacity ≈ 2350/0.80 ≈ 2937 W
# Rounded conservatively to 3000 Wp = 3.0 kWc
ESTIMATED_CAPACITY_WP   = 3000.0           # Watts-peak (estimated, NOT measured nameplate)
ESTIMATED_CAPACITY_KWC  = ESTIMATED_CAPACITY_WP / 1000.0  # 3.0 kWc
CAPACITY_ESTIMATION_NOTE = (
    "Estimated from observed max power (~2350 W) at near-STC irradiance, "
    "assuming PR=0.80. NOT a measured nameplate value. Uncertainty ±10%."
)


# ── Raw column mapping ────────────────────────────────────────────────────────
RAW_COLUMNS = {
    "Unnamed: 0":       "timestamp",
    "Power":            "power_w",
    "Irradiation":      "ghi_wm2",
    "Temperature":      "temperature_2m_c",
    "Cell_temperature": "cell_temperature_c",
    "Humidity":         "relative_humidity_pct",
    "Wind_speed":       "wind_speed_ms",
    "Pressure":         "pressure_hpa",
}

# Required output schema columns
VALIDATION_SCHEMA = [
    "timestamp",
    "power_w",
    "power_kw",
    "power_normalized",          # power_w / ESTIMATED_CAPACITY_WP (0–1 scale)
    "ghi_wm2",
    "temperature_2m_c",
    "cell_temperature_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "pressure_hpa",
    "latitude",
    "longitude",
    "altitude_m",
    "estimated_capacity_wp",
    "capacity_estimation_note",
    "production_source",
    "location_source",
    "original_resolution_min",
    "resampled_to_min",
    "night_flag",                # 1 = dark (ghi_wm2 == 0 or near 0)
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load and rename the raw ENSTAB CSV. Returns the original 5-min series."""
    df = pd.read_csv(path)
    df = df.rename(columns=RAW_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Run data quality checks and return a summary dict."""
    diffs = df["timestamp"].diff().dropna()
    dominant_freq = diffs.mode().iloc[0]

    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "time_start": str(df["timestamp"].min()),
        "time_end": str(df["timestamp"].max()),
        "dominant_resolution": str(dominant_freq),
        "n_duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "n_missing_any": int(df.isnull().sum().sum()),
        "n_missing_per_col": df.isnull().sum().to_dict(),
        "n_negative_power": int((df["power_w"] < 0).sum()),
        "power_min_w": float(df["power_w"].min()),
        "power_max_w": float(df["power_w"].max()),
        "power_mean_w": float(df["power_w"].mean()),
        "ghi_max_wm2": float(df["ghi_wm2"].max()),
        "estimated_capacity_wp": ESTIMATED_CAPACITY_WP,
        "capacity_estimation_note": CAPACITY_ESTIMATION_NOTE,
        "production_source": PRODUCTION_SOURCE,
    }
    return report


def add_solar_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Add pvlib solar position + clear-sky for Borj Cedria location."""
    try:
        import pvlib
        times = pd.DatetimeIndex(df["timestamp"]).tz_localize(
            "Africa/Tunis", ambiguous="NaT", nonexistent="NaT"
        )
        valid = ~times.isna()
        solpos = pvlib.solarposition.get_solarposition(
            times[valid], ENSTAB_LATITUDE, ENSTAB_LONGITUDE,
            altitude=ENSTAB_ALTITUDE_M
        )
        df = df.copy()
        df["solar_elevation_deg"] = np.nan
        df["solar_azimuth_deg"]   = np.nan
        df["solar_zenith_deg"]    = np.nan
        df["daylight_flag"]       = 0
        df.loc[valid, "solar_elevation_deg"] = solpos["apparent_elevation"].values
        df.loc[valid, "solar_azimuth_deg"]   = solpos["azimuth"].values
        df.loc[valid, "solar_zenith_deg"]    = solpos["apparent_zenith"].values
        df.loc[valid, "daylight_flag"]       = (solpos["apparent_elevation"].values > 0).astype(int)

        # Clear-sky GHI for context
        linke = pvlib.clearsky.lookup_linke_turbidity(times[valid], ENSTAB_LATITUDE, ENSTAB_LONGITUDE)
        airm_rel = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])
        pressure = pvlib.atmosphere.alt2pres(ENSTAB_ALTITUDE_M)
        airm_abs = pvlib.atmosphere.get_absolute_airmass(airm_rel, pressure)
        cs = pvlib.clearsky.ineichen(solpos["apparent_zenith"], airm_abs, linke, altitude=ENSTAB_ALTITUDE_M)
        df.loc[valid, "clearsky_ghi_wm2"] = cs["ghi"].values
        df["clearsky_ghi_wm2"] = df.get("clearsky_ghi_wm2", pd.Series(np.nan, index=df.index))
        df["clearsky_index"] = np.where(
            df["clearsky_ghi_wm2"] > 10,
            np.clip(df["ghi_wm2"] / df["clearsky_ghi_wm2"], 0, 1.5),
            np.nan,
        )
    except ImportError:
        warnings.warn("pvlib not available — skipping solar geometry for ENSTAB data")
        df["solar_elevation_deg"] = np.nan
        df["solar_azimuth_deg"]   = np.nan
        df["solar_zenith_deg"]    = np.nan
        df["daylight_flag"]       = np.where(df["ghi_wm2"] > 5, 1, 0)
        df["clearsky_ghi_wm2"]    = np.nan
        df["clearsky_index"]      = np.nan
    return df


def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 5-minute ENSTAB data to 15-minute resolution.

    Energy-preserving resampling:
      - power_w: mean over the 3×5-min sub-intervals (mean power → same energy
        when multiplied by same interval length)
      - irradiance/weather: mean (instantaneous-style variables)
      - night_flag: any(flag == 1) within the 15-min window

    Labels the resampled series with resampled_to_min=15.
    """
    df = df.set_index("timestamp")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    meta_cols = [c for c in df.columns if c not in numeric_cols]

    resampled = df[numeric_cols].resample("15min").mean()
    for col in meta_cols:
        resampled[col] = df[col].resample("15min").first()

    resampled["original_resolution_min"] = 5
    resampled["resampled_to_min"] = 15
    resampled = resampled.reset_index()
    return resampled


def prepare_validation_dataset(
    raw_path: str | Path,
    out_path: str | Path | None = None,
    resample_to_15min: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: load → rename → quality check → solar geometry →
    optional resample → attach provenance → save.

    Parameters
    ----------
    raw_path : path to the raw enstab_borj_cedria.csv
    out_path : optional output path for the cleaned CSV
    resample_to_15min : if True (default), resample 5-min data to 15-min

    Returns
    -------
    DataFrame with VALIDATION_SCHEMA + solar geometry columns
    """
    print(f"[ENSTAB] Loading raw data from {raw_path} ...")
    df = load_raw(raw_path)

    report = quality_report(df)
    print(f"[ENSTAB] Quality report:")
    for k, v in report.items():
        if k != "n_missing_per_col":
            print(f"  {k}: {v}")

    # Derived columns
    df["power_kw"] = df["power_w"] / 1000.0
    df["power_normalized"] = np.clip(df["power_w"] / ESTIMATED_CAPACITY_WP, 0, None)
    df["night_flag"] = (df["ghi_wm2"] <= 5.0).astype(int)
    df["latitude"]   = ENSTAB_LATITUDE
    df["longitude"]  = ENSTAB_LONGITUDE
    df["altitude_m"] = ENSTAB_ALTITUDE_M
    df["estimated_capacity_wp"]    = ESTIMATED_CAPACITY_WP
    df["capacity_estimation_note"] = CAPACITY_ESTIMATION_NOTE
    df["production_source"]        = PRODUCTION_SOURCE
    df["location_source"]          = LOCATION_SOURCE
    df["original_resolution_min"]  = 5
    df["resampled_to_min"]         = 5  # updated after resample

    print("[ENSTAB] Adding solar geometry (pvlib) ...")
    df = add_solar_geometry(df)

    if resample_to_15min:
        print("[ENSTAB] Resampling 5-min → 15-min (mean power, energy-preserving) ...")
        df = _resample_15min_internal(df)

    # Enforce non-negative power (physical constraint)
    df["power_w"] = np.clip(df["power_w"], 0, None)
    df["power_kw"] = np.clip(df["power_kw"], 0, None)
    df["power_normalized"] = np.clip(df["power_normalized"], 0, None)

    # Enforce night = 0 (physical constraint — if solar_elevation <= 0)
    night_physical = df.get("solar_elevation_deg", pd.Series(np.nan, index=df.index)) <= 0
    df.loc[night_physical, "power_w"] = 0.0
    df.loc[night_physical, "power_kw"] = 0.0
    df.loc[night_physical, "power_normalized"] = 0.0
    df.loc[night_physical, "night_flag"] = 1

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[ENSTAB] Saved {len(df):,} rows to {out_path}")

    return df


def _resample_15min_internal(df: pd.DataFrame) -> pd.DataFrame:
    """Internal resample helper operating on a df with timestamp column."""
    df = df.set_index("timestamp")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Preserve first value for categorical/string meta columns
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    resampled_num = df[numeric_cols].resample("15min").mean()
    resampled_meta = df[meta_cols].resample("15min").first()
    result = pd.concat([resampled_num, resampled_meta], axis=1).reset_index()
    result["original_resolution_min"] = 5
    result["resampled_to_min"] = 15
    return result


if __name__ == "__main__":
    import sys
    raw = Path("data/external/enstab_borj_cedria.csv")
    out = Path("data/validation/enstab_borj_cedria_real.csv")
    if not raw.exists():
        print(f"ERROR: {raw} not found", file=sys.stderr)
        sys.exit(1)
    df = prepare_validation_dataset(raw, out_path=out, resample_to_15min=True)
    print(f"\nFinal dataset: {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"production_source: {df['production_source'].iloc[0]}")
    print(f"Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\nSample (daytime):")
    print(df[df["ghi_wm2"] > 100][["timestamp","power_w","power_normalized","ghi_wm2","solar_elevation_deg"]].head(8).to_string(index=False))
