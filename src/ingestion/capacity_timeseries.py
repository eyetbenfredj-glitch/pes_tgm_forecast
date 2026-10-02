"""
Build P_capacity(district, t): a time-dependent installed-capacity signal
per district, derived from the 3 official Prosol snapshots.

Strategy (documented, transparent - CDC section 3):
  - t < first snapshot (2025-12-01):  hold first known value (no earlier data)
  - between two snapshots:            linear interpolation on power_mw_since_2011
  - t > last snapshot (2026-07-01):   hold last known value

Designed so a 4th, 5th, ... monthly snapshot can be appended later without any
change to this module or to the ML architecture: it always interpolates
between whatever consecutive snapshots bracket `t`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_capacity_timeseries(
    fleet_df: pd.DataFrame,
    freq: str = "15min",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Returns a long dataframe: district_id, district, governorate, latitude,
    longitude, timestamp, capacity_mw, n_installations, capacity_source.

    capacity_source in {"held_before_first", "interpolated", "held_after_last"}
    so every value stays traceable (CDC Rule 8: every prediction traceable to
    its inputs - this extends to every derived feature).
    """
    fleet_df = fleet_df.sort_values(["district_id", "snapshot_date"]).copy()
    snap_dates = sorted(fleet_df["snapshot_date"].unique())
    t0, t1 = snap_dates[0], snap_dates[-1]

    if start is None:
        start = t0
    if end is None:
        end = t1
    idx = pd.date_range(start=start, end=end, freq=freq)

    out_frames = []
    for district_id, g in fleet_df.groupby("district_id"):
        g = g.sort_values("snapshot_date")
        meta = g.iloc[0]
        snap_t = g["snapshot_date"].to_numpy()
        snap_cap = g["power_mw_since_2011"].to_numpy(dtype=float)
        snap_n = g["installations_since_2011"].to_numpy(dtype=float)

        # numeric time axis (seconds since epoch) for interpolation
        idx_num = idx.astype("int64").to_numpy()
        snap_t_num = snap_t.astype("int64")

        cap = np.interp(idx_num, snap_t_num, snap_cap)
        n_inst = np.interp(idx_num, snap_t_num, snap_n)

        source = np.full(len(idx), "interpolated", dtype=object)
        source[idx < snap_t.min()] = "held_before_first"
        source[idx > snap_t.max()] = "held_after_last"

        out_frames.append(pd.DataFrame({
            "district_id": district_id,
            "district": meta["district"],
            "governorate": meta["governorate"],
            "latitude": meta["latitude"],
            "longitude": meta["longitude"],
            "timestamp": idx,
            "capacity_mw": cap,
            "n_installations": n_inst,
            "capacity_source": source,
        }))

    result = pd.concat(out_frames, ignore_index=True)
    return result


def capacity_growth_features(cap_ts: pd.DataFrame) -> pd.DataFrame:
    """Adds capacity_growth_mw_30d and capacity_growth_rate_pct_30d per district."""
    cap_ts = cap_ts.sort_values(["district_id", "timestamp"]).copy()
    out = []
    for did, g in cap_ts.groupby("district_id"):
        g = g.set_index("timestamp")
        g["capacity_growth_mw_30d"] = g["capacity_mw"] - g["capacity_mw"].shift(freq="30D")
        g["capacity_growth_rate_pct_30d"] = (
            100 * g["capacity_growth_mw_30d"] / g["capacity_mw"].shift(freq="30D")
        )
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    from src.ingestion.fleet_loader import load_fleet_csv
    df = load_fleet_csv("data/raw/pv_fleet_prosol_3_snapshots.csv")
    cap_ts = build_capacity_timeseries(df, freq="1D")  # daily for a quick sanity check
    print(cap_ts.head(10))
    print(cap_ts["capacity_source"].value_counts())
    print("Districts:", cap_ts["district_id"].nunique(), "| Rows:", len(cap_ts))
