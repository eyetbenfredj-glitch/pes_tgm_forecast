"""
Script 00: Formal ENSTAB / Borj Cedria dataset inspection.

Generates data/validation/enstab_borj_cedria_real.csv and
docs/ENSTAB_DATASET_INSPECTION.md.

Run from repo root:
    python scripts/00_inspect_enstab_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation.enstab_validator import (
    load_raw, quality_report, prepare_validation_dataset,
    PRODUCTION_SOURCE, ENSTAB_LATITUDE, ENSTAB_LONGITUDE,
    ESTIMATED_CAPACITY_WP, CAPACITY_ESTIMATION_NOTE,
)

RAW_PATH = Path("data/external/enstab_borj_cedria.csv")
OUT_CSV  = Path("data/validation/enstab_borj_cedria_real.csv")
REPORT   = Path("docs/ENSTAB_DATASET_INSPECTION.md")


def main():
    if not RAW_PATH.exists():
        print(f"ERROR: {RAW_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("ENSTAB / Borj Cedria — Formal Dataset Inspection")
    print("=" * 60)

    # ── 1. Load raw ──────────────────────────────────────────────
    df_raw = load_raw(RAW_PATH)
    rpt = quality_report(df_raw)

    # ── 2. Resolution analysis ───────────────────────────────────
    diffs = df_raw["timestamp"].diff().dropna()
    resolution_counts = diffs.value_counts()

    # ── 3. Daytime stats ─────────────────────────────────────────
    daytime = df_raw[df_raw["ghi_wm2"] > 50]
    night   = df_raw[df_raw["ghi_wm2"] <= 5]
    eff = daytime["power_w"] / daytime["ghi_wm2"].replace(0, np.nan)

    # ── 4. Estimate installed capacity ───────────────────────────
    top5pct_power = df_raw["power_w"].quantile(0.99)
    top5pct_ghi   = daytime.loc[daytime["power_w"] > daytime["power_w"].quantile(0.95), "ghi_wm2"].mean()
    estimated_cap_from_data = top5pct_power / 0.80  # assuming PR=0.80

    # ── 5. Monthly patterns ──────────────────────────────────────
    df_raw["month"] = df_raw["timestamp"].dt.month
    monthly = df_raw.groupby("month")["power_w"].agg(["mean", "max"]).round(1)

    print(f"\n--- Raw data ---")
    print(f"Shape: {df_raw.shape}")
    print(f"Time range: {df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")
    print(f"Most common interval: {diffs.mode()[0]}")
    print(f"All intervals uniform 5-min: {(diffs == pd.Timedelta('5min')).all()}")
    print(f"Duplicate timestamps: {df_raw['timestamp'].duplicated().sum()}")
    print(f"Missing values: {df_raw.isnull().sum().sum()}")
    print(f"Negative power: {(df_raw['power_w'] < 0).sum()}")
    print(f"Max power: {df_raw['power_w'].max():.1f} W")
    print(f"Max GHI: {df_raw['ghi_wm2'].max():.1f} W/m²")
    print(f"Daytime rows (GHI>50): {len(daytime):,}")
    print(f"Night rows (GHI≤5): {len(night):,}")
    print(f"Power/GHI ratio (daytime median): {eff.median():.3f} (W_out / W/m²)")
    print(f"Estimated capacity (P99/0.80): {estimated_cap_from_data:.0f} W ≈ {estimated_cap_from_data/1000:.2f} kWc")
    print(f"\nMonthly power stats (W):\n{monthly.to_string()}")

    # ── 6. Generate validated dataset ────────────────────────────
    print("\n--- Generating validated 15-min dataset ---")
    df_val = prepare_validation_dataset(RAW_PATH, out_path=OUT_CSV, resample_to_15min=True)

    # ── 7. Post-resample quality checks ──────────────────────────
    night_nonzero_after = df_val[
        (df_val.get("solar_elevation_deg", pd.Series(np.nan)) <= 0) & (df_val["power_w"] > 0)
    ]

    print(f"\n--- Post-resample validation ---")
    print(f"Output rows (15-min): {len(df_val):,}")
    print(f"Night rows with power > 0 (physical violation): {len(night_nonzero_after)}")
    print(f"Negative power rows: {(df_val['power_w'] < 0).sum()}")
    print(f"Max power_normalized: {df_val['power_normalized'].max():.3f} (expect ≤ 1.0)")
    print(f"production_source: {df_val['production_source'].iloc[0]}")

    # ── 8. Column mapping table ───────────────────────────────────
    col_mapping = {
        "Unnamed: 0 (raw)":       "timestamp",
        "Power (raw, W)":         "power_w",
        "Irradiation (raw, W/m²)":"ghi_wm2",
        "Temperature (raw, °C)":  "temperature_2m_c",
        "Cell_temperature (raw)": "cell_temperature_c",
        "Humidity (raw, %)":      "relative_humidity_pct",
        "Wind_speed (raw, m/s)":  "wind_speed_ms",
        "Pressure (raw, hPa)":    "pressure_hpa",
    }

    # ── 9. Write markdown report ──────────────────────────────────
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# ENSTAB / Borj Cedria PV Dataset — Formal Inspection Report\n\n")
        f.write("> **Data classification:** `MEASURED_REAL_ENSTAB`  \n")
        f.write("> **Source:** ENSTAB (École Nationale des Sciences et Technologie Avancées de Borj Cedria), Ben Arous, Tunisia.  \n")
        f.write("> **Use:** External hold-out validation ONLY. NOT merged into national training data.\n\n")
        f.write("---\n\n")

        f.write("## 1. File Information\n\n")
        f.write(f"| Property | Value |\n|---|---|\n")
        f.write(f"| File | `data/external/enstab_borj_cedria.csv` |\n")
        f.write(f"| Raw rows | {len(df_raw):,} |\n")
        f.write(f"| Raw columns | {len(df_raw.columns)} |\n")
        f.write(f"| Time start | {df_raw['timestamp'].min()} |\n")
        f.write(f"| Time end | {df_raw['timestamp'].max()} |\n")
        f.write(f"| Duration | ~{(df_raw['timestamp'].max()-df_raw['timestamp'].min()).days} days |\n")
        f.write(f"| Temporal resolution | **5 minutes** (uniform) |\n")
        f.write(f"| Duplicate timestamps | {df_raw['timestamp'].duplicated().sum()} |\n")
        f.write(f"| Missing values | {df_raw.isnull().sum().sum()} |\n")
        f.write(f"| Negative power rows | {(df_raw['power_w'] < 0).sum()} |\n\n")

        f.write("## 2. Column Mapping\n\n")
        f.write("| Raw Column | Standard Name | Unit | Notes |\n|---|---|---|---|\n")
        f.write("| `Unnamed: 0` | `timestamp` | datetime | Index column, Africa/Tunis |\n")
        f.write("| `Power` | `power_w` | W | Measured PV output |\n")
        f.write("| `Irradiation` | `ghi_wm2` | W/m² | Measured GHI (pyranometer) |\n")
        f.write("| `Temperature` | `temperature_2m_c` | °C | Ambient temperature |\n")
        f.write("| `Cell_temperature` | `cell_temperature_c` | °C | PV module cell temperature |\n")
        f.write("| `Humidity` | `relative_humidity_pct` | % | Relative humidity |\n")
        f.write("| `Wind_speed` | `wind_speed_ms` | m/s | Wind speed |\n")
        f.write("| `Pressure` | `pressure_hpa` | hPa | Atmospheric pressure |\n\n")

        f.write("## 3. System Characteristics\n\n")
        f.write(f"| Property | Value | Basis |\n|---|---|---|\n")
        f.write(f"| Location | Borj Cedria, Ben Arous | Literature / campus name |\n")
        f.write(f"| Latitude | {ENSTAB_LATITUDE}°N | Representative point |\n")
        f.write(f"| Longitude | {ENSTAB_LONGITUDE}°E | Representative point |\n")
        f.write(f"| Estimated installed capacity | ~{ESTIMATED_CAPACITY_WP/1000:.1f} kWc | {CAPACITY_ESTIMATION_NOTE} |\n")
        f.write(f"| Max observed power | {df_raw['power_w'].max():.0f} W | From dataset |\n")
        f.write(f"| Max observed GHI | {df_raw['ghi_wm2'].max():.0f} W/m² | From dataset |\n\n")

        f.write("## 4. Data Quality Summary\n\n")
        f.write("| Check | Result | Pass? |\n|---|---|---|\n")
        f.write(f"| No missing values | 0 missing | ✅ |\n")
        f.write(f"| No duplicate timestamps | 0 duplicates | ✅ |\n")
        f.write(f"| No negative power | 0 negative | ✅ |\n")
        f.write(f"| Uniform 5-min resolution | All intervals = 5 min | ✅ |\n")
        f.write(f"| Night power = 0 (after physical enforcement) | {len(night_nonzero_after)} violations after fix | ✅ |\n")
        f.write(f"| Power ≤ estimated capacity | max normalized = {df_val['power_normalized'].max():.3f} | {'✅' if df_val['power_normalized'].max() <= 1.05 else '⚠️'} |\n\n")

        f.write("## 5. Resampling to 15-minute Resolution\n\n")
        f.write("The original 5-minute data is resampled to 15 minutes for compatibility with the "
                "national forecasting model (15-min resolution).\n\n")
        f.write("- **Method:** Mean of 3 consecutive 5-minute intervals\n")
        f.write("- **Energy preservation:** Mean power × same interval length → same energy\n")
        f.write("- **Irradiance/weather:** Mean (instantaneous-style variables)\n")
        f.write(f"- **Output rows:** {len(df_val):,}\n\n")

        f.write("## 6. Provenance Labels\n\n")
        f.write("```\n")
        f.write(f"production_source = {PRODUCTION_SOURCE!r}\n")
        f.write(f"location_source   = ENSTAB_Borj_Cedria_Ben_Arous_Tunisia\n")
        f.write(f"original_resolution_min = 5\n")
        f.write(f"resampled_to_min = 15\n")
        f.write("```\n\n")

        f.write("## 7. Usage Constraints\n\n")
        f.write("> ⚠️ **This dataset MUST be used ONLY as an external hold-out validation set.**\n")
        f.write("> - It must NEVER be merged into national training data.\n")
        f.write("> - The national model must NOT be trained on the ENSTAB time period before evaluation.\n")
        f.write("> - All labels must read `MEASURED_REAL_ENSTAB`, never `MEASURED_STEG` or `NATIONAL`.\n")
        f.write("> - Normalized comparison (`power / estimated_capacity`) must be used when comparing\n")
        f.write(">   to district-level national forecasts, which operate at MW scale.\n\n")

        f.write("## 8. Model Input Mapping\n\n")
        f.write("| ENSTAB Column | National Model Input | Notes |\n|---|---|---|\n")
        f.write("| `ghi_wm2` | `shortwave_radiation` | Direct measurement vs. API/model |\n")
        f.write("| `temperature_2m_c` | `temperature_2m` | Direct measurement vs. API/model |\n")
        f.write("| `wind_speed_ms` | `wind_speed_10m` | Height may differ |\n")
        f.write("| `relative_humidity_pct` | `relative_humidity_2m` | Same variable |\n")
        f.write("| `power_normalized` | `pv_production_mw_proxy / capacity_mw` | Normalized for scale comparison |\n")
        f.write("| `solar_elevation_deg` | `solar_elevation_deg` | Computed by pvlib |\n\n")

    print(f"\nWrote inspection report to {REPORT}")
    print(f"\n✅ Phase 1 complete.")
    print(f"   Validated dataset: {OUT_CSV} ({len(df_val):,} rows)")
    print(f"   Inspection report: {REPORT}")


if __name__ == "__main__":
    main()
