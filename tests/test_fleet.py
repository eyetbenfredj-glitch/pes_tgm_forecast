import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.fleet_loader import load_fleet_csv, validate_fleet


def test_fleet_loads_and_validates():
    df = load_fleet_csv("data/raw/pv_fleet_prosol_3_snapshots.csv")
    report = validate_fleet(df)
    assert report.n_districts == 50
    assert report.duplicate_keys == 0
    assert report.missing_coordinates == 0
    assert report.is_usable


def test_fleet_no_negative_capacity():
    df = load_fleet_csv("data/raw/pv_fleet_prosol_3_snapshots.csv")
    assert (df["power_mw_since_2011"] >= 0).all()
