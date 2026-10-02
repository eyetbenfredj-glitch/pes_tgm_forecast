import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation.physical_checks import run_all_checks
from src.physics.pv_model import generate_proxy_production, physics_pv_output_mw


def test_physics_output_never_exceeds_capacity():
    import numpy as np
    cap = np.array([10.0] * 5)
    out = physics_pv_output_mw(
        capacity_mw=cap,
        ghi_wm2=np.array([0, 300, 600, 900, 1200]),  # 1200 > STC on purpose (edge case)
        clearsky_ghi_wm2=np.array([0, 300, 600, 900, 1000]),
        ambient_temp_c=np.array([15, 20, 25, 35, 45]),
        performance_ratio=0.8,
    )
    assert (out <= cap + 1e-9).all()
    assert (out >= 0).all()


def test_physics_output_zero_at_night():
    import numpy as np
    out = physics_pv_output_mw(
        capacity_mw=np.array([10.0]), ghi_wm2=np.array([0.0]),
        clearsky_ghi_wm2=np.array([0.0]), ambient_temp_c=np.array([18.0]),
    )
    assert out[0] == 0.0


def test_proxy_production_respects_bounds_and_night():
    n = 6
    df = pd.DataFrame({
        "district_id": ["D01"] * n,
        "capacity_mw": [10.0] * n,
        "shortwave_radiation": [0, 100, 500, 900, 400, 0],
        "clearsky_ghi_wm2": [0, 150, 550, 950, 450, 0],
        "temperature_2m": [15, 18, 25, 32, 28, 16],
        "daylight_flag": [0, 1, 1, 1, 1, 0],
    })
    out = generate_proxy_production(df, noise_std_fraction=0.1)  # high noise to stress-test clipping
    assert (out["pv_production_mw_proxy"] >= 0).all()
    assert (out["pv_production_mw_proxy"] <= out["capacity_mw"] + 1e-9).all()
    night = out["daylight_flag"] == 0
    assert (out.loc[night, "pv_production_mw_proxy"] == 0).all()


def test_run_all_checks_flags_violations():
    df = pd.DataFrame({
        "capacity_mw": [10, 10],
        "target": [0, 15],  # second row impossibly exceeds capacity
        "daylight_flag": [0, 1],
        "solar_elevation_deg": [-5, 30],
    })
    out = run_all_checks(df, production_col="target")
    assert out["check_bounds_ok"].iloc[1] == False  # noqa: E712
    assert out["physical_anomaly_flag"].iloc[1] == True  # noqa: E712
