"""
Builds data/processed/weather_historical.csv for every district in the fleet.

Uses the configured WeatherProvider (configs/config.yaml -> weather.provider).
Falls back automatically to SyntheticClearSkyProvider if the real provider
raises (e.g. no outbound internet access to open-meteo.com in this sandbox),
logging the fallback explicitly so it is never silently mistaken for real data.

Usage (run from repo root):
    python scripts/02_build_weather_dataset.py --start 2026-06-01 --end 2026-08-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.fleet_loader import load_fleet_csv
from src.ingestion.weather_provider import get_provider, SyntheticClearSkyProvider
from src.ingestion.data_mode import get_data_mode, DataMode, RealDataUnavailableError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-08-01")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="data/processed/weather_historical.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    fleet = load_fleet_csv(cfg["paths"]["raw_fleet_csv"])
    districts = fleet.drop_duplicates("district_id")[
        ["district_id", "district", "governorate", "latitude", "longitude"]
    ]

    provider_name = cfg["weather"]["provider"]
    provider = get_provider(provider_name)
    mode = get_data_mode()
    fallback_used = False

    frames = []
    for _, row in districts.iterrows():
        try:
            df = provider.get_historical_weather(
                row["district_id"], row["district"], row["governorate"],
                row["latitude"], row["longitude"], args.start, args.end,
            )
        except Exception as e:  # noqa: BLE001
            if mode == DataMode.REAL:
                raise RealDataUnavailableError(
                    f"real historical weather ({provider_name})",
                    detail=f"First failure on district {row['district_id']} ({row['district']}): {e}",
                ) from e
            if not fallback_used:
                print(f"[WARN] DATA_MODE=demo: real provider '{provider_name}' unreachable ({e}). "
                      f"Falling back to {cfg['weather']['mode_fallback']} for ALL districts, tagged "
                      f"SYNTHETIC_CLEARSKY. See docs/DATA_READINESS_REPORT.md. "
                      f"(This fallback is FORBIDDEN when DATA_MODE=real.)")
            fallback_used = True
            provider = SyntheticClearSkyProvider()
            df = provider.get_historical_weather(
                row["district_id"], row["district"], row["governorate"],
                row["latitude"], row["longitude"], args.start, args.end,
            )
        frames.append(df)

    weather = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(args.out, index=False)
    print(f"DATA_MODE={mode.value} | Wrote {len(weather):,} rows for {districts.shape[0]} districts -> {args.out}")
    print(f"weather_source values: {weather['weather_source'].unique().tolist()}")
    print(f"Real data used: {not fallback_used}")


if __name__ == "__main__":
    main()
