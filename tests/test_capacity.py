import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.fleet_loader import load_fleet_csv
from src.ingestion.capacity_timeseries import build_capacity_timeseries


def _cap_ts():
    df = load_fleet_csv("data/raw/pv_fleet_prosol_3_snapshots.csv")
    return build_capacity_timeseries(df, freq="1D")


def test_capacity_non_negative():
    cap = _cap_ts()
    assert (cap["capacity_mw"] >= 0).all()


def test_capacity_covers_all_districts():
    cap = _cap_ts()
    assert cap["district_id"].nunique() == 50


def test_capacity_source_traceable():
    cap = _cap_ts()
    assert set(cap["capacity_source"].unique()) <= {"interpolated", "held_before_first", "held_after_last"}


def test_capacity_matches_snapshot_at_snapshot_dates():
    """At the exact snapshot dates, interpolated capacity must equal the raw snapshot value."""
    fleet = load_fleet_csv("data/raw/pv_fleet_prosol_3_snapshots.csv")
    cap = build_capacity_timeseries(fleet, freq="1D")
    for _, row in fleet.iterrows():
        match = cap[(cap["district_id"] == row["district_id"]) & (cap["timestamp"] == row["snapshot_date"])]
        assert len(match) == 1
        assert abs(match["capacity_mw"].iloc[0] - row["power_mw_since_2011"]) < 1e-6
