"""
Fleet loader & validator for the Prosol PV fleet snapshots.

Reads pv_fleet_prosol_3_snapshots.csv (3 official snapshots: Dec-2025, Mar-2026,
Jul-2026) and validates schema, coordinate consistency, duplicate keys and
capacity totals.

IMPORTANT (per CDC section 2-3):
The three snapshots are FLEET STATES, not independent ML observations.
This module only loads & validates them. Time-series capacity construction
happens in capacity_timeseries.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "snapshot_date", "snapshot_month", "district_id", "district", "governorate",
    "latitude", "longitude", "coordinate_type",
    "installations_since_2011", "power_mw_since_2011", "avg_installation_size_kwc",
    "national_installations", "national_power_mw",
    "installation_share_national_pct", "power_share_national_pct",
    "new_installations_since_previous_snapshot", "new_power_mw_since_previous_snapshot",
    "source",
]


@dataclass
class FleetValidationReport:
    n_rows: int = 0
    n_districts: int = 0
    n_governorates: int = 0
    snapshot_dates: list = field(default_factory=list)
    missing_columns: list = field(default_factory=list)
    duplicate_keys: int = 0
    missing_coordinates: int = 0
    districts_with_inconsistent_coords: list = field(default_factory=list)
    capacity_sum_vs_national_mw: dict = field(default_factory=dict)  # snapshot -> (sum_district, national_reported, diff)
    negative_or_null_capacity_rows: int = 0
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return not self.errors


def load_fleet_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def validate_fleet(df: pd.DataFrame) -> FleetValidationReport:
    report = FleetValidationReport()
    report.n_rows = len(df)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.missing_columns = missing_cols
    if missing_cols:
        report.errors.append(f"Missing required columns: {missing_cols}")
        return report  # cannot validate further safely

    report.n_districts = df["district_id"].nunique()
    report.n_governorates = df["governorate"].nunique()
    report.snapshot_dates = sorted(df["snapshot_date"].dt.strftime("%Y-%m-%d").unique().tolist())

    # Duplicate (district_id, snapshot_date) keys
    dup = df.duplicated(subset=["district_id", "snapshot_date"]).sum()
    report.duplicate_keys = int(dup)
    if dup:
        report.errors.append(f"{dup} duplicate (district_id, snapshot_date) rows found")

    # Missing coordinates
    missing_coords = df[["latitude", "longitude"]].isna().any(axis=1).sum()
    report.missing_coordinates = int(missing_coords)
    if missing_coords:
        report.errors.append(f"{missing_coords} rows with missing latitude/longitude")

    # Coordinate consistency per district across snapshots
    coord_nunique = df.groupby("district_id")[["latitude", "longitude"]].nunique()
    inconsistent = coord_nunique[(coord_nunique["latitude"] > 1) | (coord_nunique["longitude"] > 1)]
    report.districts_with_inconsistent_coords = inconsistent.index.tolist()
    if len(inconsistent):
        report.warnings.append(
            f"{len(inconsistent)} district(s) report different coordinates across snapshots: "
            f"{inconsistent.index.tolist()}"
        )

    # Negative / null capacity sanity check
    neg = (df["power_mw_since_2011"] < 0).sum() + df["power_mw_since_2011"].isna().sum()
    report.negative_or_null_capacity_rows = int(neg)
    if neg:
        report.warnings.append(f"{neg} rows with negative/null power_mw_since_2011")

    # Sum(district capacity) vs reported national_power_mw, per snapshot
    for snap, g in df.groupby(df["snapshot_date"].dt.strftime("%Y-%m-%d")):
        sum_district = round(float(g["power_mw_since_2011"].sum()), 3)
        national_reported = float(g["national_power_mw"].iloc[0])
        diff = round(sum_district - national_reported, 3)
        report.capacity_sum_vs_national_mw[snap] = {
            "sum_of_districts_mw": sum_district,
            "national_power_mw_column": national_reported,
            "diff_mw": diff,
            "diff_pct": round(100 * diff / national_reported, 2) if national_reported else None,
        }
        if abs(diff) > 0.05 * national_reported:
            report.warnings.append(
                f"Snapshot {snap}: sum of district capacities ({sum_district} MW) differs from "
                f"reported national_power_mw ({national_reported} MW) by {diff} MW "
                f"({report.capacity_sum_vs_national_mw[snap]['diff_pct']}%). "
                f"The dataset appears to cover only the districts supplied here; "
                f"national_power_mw likely reflects the FULL national Prosol fleet, "
                f"which may include districts not present in this extract."
            )

    return report


if __name__ == "__main__":
    import json
    df = load_fleet_csv("data/raw/pv_fleet_prosol_3_snapshots.csv")
    rep = validate_fleet(df)
    print(json.dumps(rep.__dict__, indent=2, default=str))
