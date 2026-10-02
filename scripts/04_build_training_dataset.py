"""
Builds the final SYNCHRONIZED, ML-ready dataset:

    timestamp | district_id | district | governorate | capacity_mw |
    weather features | solar geometry features | pv_production_mw_proxy | ...

at the configured resolution (default 15 minutes), by:
  1. building the capacity time series from the 3 Prosol snapshots,
  2. loading the hourly weather dataset (built by 02_build_weather_dataset.py)
     and upsampling it to 15-min via linear interpolation (explicitly flagged:
     this is a RESOLUTION change, not new INFORMATION - CDC section 9),
  3. computing solar geometry at the target resolution,
  4. running the physics-based proxy production generator (CDC section 6/7/35).

Also writes a minimal "simple view" CSV matching the exact format requested
for downstream model training:  timestamp, district, pv_production_mw

Usage (run from repo root):
    python scripts/04_build_training_dataset.py --start 2026-06-01 --end 2026-08-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.fleet_loader import load_fleet_csv
from src.ingestion.capacity_timeseries import build_capacity_timeseries
from src.features.solar_geometry import add_solar_geometry
from src.physics.pv_model import generate_proxy_production
from src.ingestion.data_mode import get_data_mode, DataMode, RealDataUnavailableError


def upsample_weather_to_resolution(weather: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Linear interpolation of hourly weather to a finer resolution, per district.
    Flags the result as resampled (not genuine higher-frequency information)."""
    out = []
    interp_cols = [
        "temperature_2m", "relative_humidity_2m", "cloud_cover",
        "shortwave_radiation", "direct_radiation", "diffuse_radiation",
        "direct_normal_irradiance", "wind_speed_10m", "precipitation",
    ]
    for did, g in weather.groupby("district_id"):
        g = g.set_index("timestamp").sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        gi = g[interp_cols].reindex(g.index.union(idx)).interpolate("time").reindex(idx)
        meta_cols = ["district", "governorate", "requested_latitude", "requested_longitude",
                     "weather_grid_latitude", "weather_grid_longitude", "weather_source"]
        for c in meta_cols:
            gi[c] = g[c].iloc[0]
        gi["district_id"] = did
        gi["weather_resolution_note"] = "resampled_from_hourly_interpolation"
        gi = gi.reset_index(names="timestamp")
        out.append(gi)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-08-01")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--weather_in", default="data/processed/weather_historical.csv")
    ap.add_argument("--out_full", default="data/processed/training_dataset_15min.csv")
    ap.add_argument("--out_simple", default="data/processed/pv_production_simple_view.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    res_min = cfg["production_target"]["resolution_minutes"]
    freq = f"{res_min}min"

    print("1/5 Loading fleet & building capacity time series...")
    fleet = load_fleet_csv(cfg["paths"]["raw_fleet_csv"])
    cap_ts = build_capacity_timeseries(fleet, freq=freq, start=args.start, end=args.end)

    print("2/5 Loading & upsampling weather...")
    weather = pd.read_csv(args.weather_in, parse_dates=["timestamp"])
    weather_hr = upsample_weather_to_resolution(weather, freq=freq)

    print("3/5 Merging capacity + weather...")
    merged = cap_ts.merge(
        weather_hr.drop(columns=["district", "governorate"]),
        on=["district_id", "timestamp"], how="inner",
    )

    print("4/5 Computing solar geometry...")
    merged = add_solar_geometry(merged, lat_col="latitude", lon_col="longitude", ts_col="timestamp")

    print("5/5 Generating physics-based DEMO/PROXY production target...")
    mode = get_data_mode()
    real_production_path = Path("data/raw/pv_production_real.csv")
    if mode == DataMode.REAL:
        if not real_production_path.exists():
            raise RealDataUnavailableError(
                "real PV production (data/raw/pv_production_real.csv)",
                detail="No real production file found. See docs/REAL_PRODUCTION_DATA_SOURCES.md - "
                       "no public real measured PV production source for Tunisia was located. "
                       "Supervised training against a real target cannot proceed under DATA_MODE=real.",
            )
        # Real production plug-in point: expects columns timestamp, district_id, production_mw
        real_prod = pd.read_csv(real_production_path, parse_dates=["timestamp"])
        merged = merged.merge(real_prod, on=["district_id", "timestamp"], how="left")
        merged = merged.rename(columns={"production_mw": "pv_production_mw_real"})
        merged["pv_production_source"] = "REAL_MEASURED"
    else:
        pt_cfg = cfg["production_target"]
        merged = generate_proxy_production(
            merged,
            performance_ratio_mean=pt_cfg["performance_ratio_mean"],
            performance_ratio_std=pt_cfg["performance_ratio_std"],
            noise_std_fraction=pt_cfg["noise_std_fraction"],
            noct_c=pt_cfg["nominal_operating_cell_temp_c"],
            temp_coeff_pct_per_c=pt_cfg["temperature_coefficient_pct_per_c"],
            random_seed=cfg["random_seed"],
        )
        assert (merged["pv_production_mw_proxy"] <= merged["capacity_mw"] + 1e-6).all(), "Production exceeds capacity!"
        assert (merged["pv_production_mw_proxy"] >= -1e-9).all(), "Negative production!"
        night_mask = merged["daylight_flag"] == 0
        assert merged.loc[night_mask, "pv_production_mw_proxy"].max() < 1e-6, "Non-zero night production!"

    Path(args.out_full).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_full, index=False)
    print(f"Full synchronized dataset: {len(merged):,} rows, {merged.shape[1]} columns -> {args.out_full}")

    prod_col = "pv_production_mw_real" if mode == DataMode.REAL else "pv_production_mw_proxy"
    simple = merged[["timestamp", "district", prod_col]].rename(
        columns={prod_col: "pv_production_mw"}
    ).sort_values(["district", "timestamp"])
    simple.to_csv(args.out_simple, index=False)
    print(f"Simple view: {len(simple):,} rows -> {args.out_simple}")
    print("\nSample (SFAX NORD, around noon):")
    sample = simple[(simple["district"].str.contains("SFAX", case=False))]
    print(sample.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
