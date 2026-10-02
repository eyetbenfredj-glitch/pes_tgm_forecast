"""
Script 10: Build ML Training Dataset with Physical Reference Target.

Takes the output of script 09 (pv_production_reference_15min.csv) and adds:
  - lag features (leakage-safe, using pv_production_mw_reference)
  - rolling features
  - all engineered temporal/solar/weather features

Target column: pv_production_mw_reference
              (replaces pv_production_mw_proxy when reference data is available)

The old proxy dataset is preserved in data/processed/training_dataset_15min.csv
for DEMO mode fallback.

Run from repo root:
    python scripts/10_build_training_dataset_real_reference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.lag_features import add_lag_features, add_rolling_features
from src.features.solar_geometry import add_solar_geometry

REF_PATH  = Path("data/processed/pv_production_reference_15min.csv")
OUT_PATH  = Path("data/processed/training_dataset_15min_real_reference.csv")
OUT_SAMPLE= Path("data/processed/training_dataset_15min_real_reference_sample.csv")
CONFIG    = Path("configs/config.yaml")

TARGET_COL = "pv_production_mw_reference"
SAMPLE_DISTRICTS = 5   # districts to include in sample CSV


def main():
    print("=" * 60)
    print("Script 10: Build Real-Reference Training Dataset")
    print("=" * 60)

    if not REF_PATH.exists():
        print(f"ERROR: {REF_PATH} not found. Run script 09 first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[1/4] Loading reference dataset from {REF_PATH} ...")
    df = pd.read_csv(REF_PATH, parse_dates=["timestamp"])
    print(f"  Rows: {len(df):,} | Districts: {df['district_id'].nunique()}")
    print(f"  Production source(s): {df['production_source'].value_counts().to_dict()}")

    # ── Validate required columns ────────────────────────────────
    required = [TARGET_COL, "district_id", "timestamp", "installed_capacity_mw"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Ensure capacity column is available under the name used by models
    if "capacity_mw" not in df.columns:
        df["capacity_mw"] = df["installed_capacity_mw"]

    # ── Ensure solar geometry is present ────────────────────────
    print("\n[2/4] Ensuring solar geometry features ...")
    solar_cols = ["solar_elevation_deg", "clearsky_ghi_wm2", "daylight_flag", "hour_sin", "hour_cos"]
    if not all(c in df.columns for c in solar_cols):
        print("  Re-computing solar geometry (not found in reference dataset) ...")
        lat_col = "latitude" if "latitude" in df.columns else None
        lon_col = "longitude" if "longitude" in df.columns else None
        if lat_col and lon_col:
            df = add_solar_geometry(df, lat_col=lat_col, lon_col=lon_col, ts_col="timestamp")
        else:
            print("  WARNING: No lat/lon columns found — solar geometry skipped.")

    # Add clear_sky_index feature
    if "clearsky_ghi_wm2" in df.columns and "ghi_wm2" in df.columns:
        df["clear_sky_index"] = np.where(
            df["clearsky_ghi_wm2"] > 10,
            np.clip(df["ghi_wm2"] / df["clearsky_ghi_wm2"], 0, 1.5),
            np.nan,
        )

    # ── Lag + rolling features on the reference target ───────────
    print("\n[3/4] Computing lag + rolling features (leakage-safe) ...")
    lag_steps = {
        "lag_1": 1,     # 15 min ago
        "lag_4": 4,     # 1 hour ago
        "lag_8": 8,     # 2 hours ago
        "lag_96": 96,   # same time yesterday
    }
    df = add_lag_features(df, TARGET_COL, group_col="district_id", lag_steps=lag_steps)
    df = add_rolling_features(df, TARGET_COL, group_col="district_id",
                               windows={"4": 4, "96": 96})

    # ── Capacity growth features ─────────────────────────────────
    df = df.sort_values(["district_id", "timestamp"])
    df["capacity_growth_30d"] = (
        df.groupby("district_id")["capacity_mw"]
        .transform(lambda s: s - s.shift(96 * 30))  # 30 days × 96 steps/day
    )

    # ── Add horizon-specific feature sets as flags ────────────────
    # These indicate which lag features are valid for each horizon
    # (intraday: all lags available; J+1/J+2/J+3: only calendar/weather)
    df["has_production_lags"] = df[[f"{TARGET_COL}_lag_1", f"{TARGET_COL}_lag_4"]].notna().all(axis=1)

    # ── Data provenance for training dataset ──────────────────────
    df["training_dataset_source"] = df["production_source"]
    df["training_target_col"] = TARGET_COL

    # ── Save ─────────────────────────────────────────────────────
    print(f"\n[4/4] Saving ...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"  ✅ Full dataset: {len(df):,} rows → {OUT_PATH}")

    # Sample: first N districts, last 30 days
    districts_sample = sorted(df["district_id"].unique())[:SAMPLE_DISTRICTS]
    t_max = df["timestamp"].max()
    t_min_sample = t_max - pd.Timedelta(days=30)
    sample = df[
        (df["district_id"].isin(districts_sample)) &
        (df["timestamp"] >= t_min_sample)
    ].copy()
    sample.to_csv(OUT_SAMPLE, index=False)
    print(f"  ✅ Sample ({SAMPLE_DISTRICTS} districts, 30 days): {len(sample):,} rows → {OUT_SAMPLE}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\nSummary:")
    print(f"  Target column: {TARGET_COL}")
    print(f"  Source labels: {df['production_source'].value_counts().to_dict()}")
    print(f"  Max {TARGET_COL}: {df[TARGET_COL].max():.4f} MW")
    print(f"  Night non-zero: {(df.get('daylight_flag', pd.Series(1)) == 0) & (df[TARGET_COL] > 1e-6)}")
    lag_col = f"{TARGET_COL}_lag_1"
    if lag_col in df.columns:
        print(f"  Lag features: {lag_col} non-null = {df[lag_col].notna().sum():,}")
    print(f"\n  Feature columns available for ML:")
    feature_cols = [c for c in df.columns if c not in [
        "timestamp", "district_id", "district", "governorate",
        "production_source", "weather_source", "fleet_snapshot_source",
        "retrieval_timestamp", "model_version", "capacity_source",
        "disaggregation_method", "weather_resolution_note",
        "training_dataset_source", "training_target_col",
        "capacity_estimation_note", "pv_calc_params", "api_endpoint",
        "pvgis_source_label", "retrieval_timestamp",
    ]]
    print(f"  {len(feature_cols)} feature columns: {feature_cols[:10]} ...")


if __name__ == "__main__":
    main()
