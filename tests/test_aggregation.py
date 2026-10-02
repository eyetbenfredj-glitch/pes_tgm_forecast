import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aggregation.hierarchical import reconcile_all_levels, check_consistency


def _toy():
    return pd.DataFrame({
        "timestamp": ["t1", "t1", "t1", "t2", "t2", "t2"],
        "district_id": ["D1", "D2", "D3", "D1", "D2", "D3"],
        "district": ["A", "B", "C", "A", "B", "C"],
        "governorate": ["G1", "G1", "G2", "G1", "G1", "G2"],
        "forecast_mw": [10.0, 5.0, 3.0, 1.0, 2.0, 3.0],
    })


def test_governorate_sum_equals_district_sum():
    levels = reconcile_all_levels(_toy(), "forecast_mw")
    g = levels["governorate"]
    g1_t1 = g[(g["timestamp"] == "t1") & (g["governorate"] == "G1")]["forecast_mw"].iloc[0]
    assert g1_t1 == 15.0  # 10 + 5


def test_national_sum_equals_governorate_sum():
    levels = reconcile_all_levels(_toy(), "forecast_mw")
    n = levels["national"]
    n_t1 = n[n["timestamp"] == "t1"]["forecast_mw"].iloc[0]
    assert n_t1 == 18.0  # 10+5+3


def test_consistency_check_passes_by_construction():
    levels = reconcile_all_levels(_toy(), "forecast_mw")
    result = check_consistency(levels, "forecast_mw")
    assert result["governorate_consistent"]
    assert result["national_consistent"]
    assert result["max_governorate_diff"] < 1e-9
    assert result["max_national_diff"] < 1e-9
