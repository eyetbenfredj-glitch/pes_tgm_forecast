"""
Physics-inspired PV baseline model (CDC section 7).

    P_pv(t) = Capacity(t) x normalized_solar_resource(t)
              x temperature_efficiency_correction(t)
              x system_performance_factor

with hard physical constraints:
    0 <= P_pv(t) <= Capacity(t)
    P_pv(t) ~= 0 at night (solar elevation <= 0)

Used for:
  - the physics baseline forecast (fallback level 3, CDC section 18)
  - feature generation
  - the labelled DEMO/PROXY production target when no real STEG production
    data is available (CDC section 6, 35) - adds bounded stochastic noise to
    emulate real-world measurement/production variability, but the process
    itself is fully physical & documented, never a black box.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REFERENCE_GHI_WM2 = 1000.0  # STC reference irradiance
NOCT_REFERENCE_GHI = 800.0
NOCT_REFERENCE_AMBIENT_C = 20.0


def normalized_solar_resource(ghi_wm2: np.ndarray, clearsky_ghi_wm2: np.ndarray) -> np.ndarray:
    """Fraction of STC irradiance actually available, floored at 0."""
    return np.clip(ghi_wm2 / REFERENCE_GHI_WM2, 0, None)


def cell_temperature_c(ambient_temp_c: np.ndarray, ghi_wm2: np.ndarray,
                        noct_c: float = 45.0) -> np.ndarray:
    """Simple NOCT model for module cell temperature."""
    return ambient_temp_c + (noct_c - NOCT_REFERENCE_AMBIENT_C) * (ghi_wm2 / NOCT_REFERENCE_GHI)


def temperature_efficiency_correction(cell_temp_c: np.ndarray, temp_coeff_pct_per_c: float = -0.40) -> np.ndarray:
    """Multiplicative correction relative to STC (25 degC)."""
    return 1 + (temp_coeff_pct_per_c / 100.0) * (cell_temp_c - 25.0)


def physics_pv_output_mw(
    capacity_mw: np.ndarray,
    ghi_wm2: np.ndarray,
    clearsky_ghi_wm2: np.ndarray,
    ambient_temp_c: np.ndarray,
    performance_ratio: float | np.ndarray = 0.80,
    noct_c: float = 45.0,
    temp_coeff_pct_per_c: float = -0.40,
) -> np.ndarray:
    """Deterministic physical PV output estimate, in MW. Vectorized."""
    resource = normalized_solar_resource(ghi_wm2, clearsky_ghi_wm2)
    t_cell = cell_temperature_c(ambient_temp_c, ghi_wm2, noct_c=noct_c)
    temp_corr = temperature_efficiency_correction(t_cell, temp_coeff_pct_per_c)
    p = capacity_mw * resource * temp_corr * performance_ratio
    return np.clip(p, 0, capacity_mw)


def generate_proxy_production(
    df: pd.DataFrame,
    performance_ratio_mean: float = 0.80,
    performance_ratio_std: float = 0.03,
    noise_std_fraction: float = 0.04,
    noct_c: float = 45.0,
    temp_coeff_pct_per_c: float = -0.40,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Builds the labelled DEMO/PROXY PV production target column
    `pv_production_mw_proxy` on top of an input dataframe that already
    contains, per row: capacity_mw, shortwave_radiation, clearsky_ghi_wm2,
    temperature_2m, daylight_flag.

    A per-district random performance ratio (soiling/inverter/wiring losses,
    static over the whole horizon) plus small heteroskedastic Gaussian noise
    emulate real fleet variability WITHOUT ever claiming to be observed data.
    Physical bounds (0 <= P <= capacity, ~0 at night) are enforced last, so
    noise can never violate them.
    """
    df = df.copy()
    rng = np.random.default_rng(random_seed)

    district_pr = {
        did: float(np.clip(rng.normal(performance_ratio_mean, performance_ratio_std), 0.55, 0.95))
        for did in df["district_id"].unique()
    }
    pr_array = df["district_id"].map(district_pr).to_numpy()

    p_clean = physics_pv_output_mw(
        capacity_mw=df["capacity_mw"].to_numpy(),
        ghi_wm2=df["shortwave_radiation"].to_numpy(),
        clearsky_ghi_wm2=df["clearsky_ghi_wm2"].to_numpy(),
        ambient_temp_c=df["temperature_2m"].to_numpy(),
        performance_ratio=pr_array,
        noct_c=noct_c,
        temp_coeff_pct_per_c=temp_coeff_pct_per_c,
    )

    noise = rng.normal(0, noise_std_fraction, len(df)) * df["capacity_mw"].to_numpy()
    p_noisy = p_clean + noise
    p_noisy = np.clip(p_noisy, 0, df["capacity_mw"].to_numpy())
    if "daylight_flag" in df.columns:
        p_noisy = np.where(df["daylight_flag"].to_numpy() == 0, 0.0, p_noisy)

    df["pv_production_mw_proxy"] = p_noisy
    df["pv_production_source"] = "DEMO_PROXY_PHYSICS_SIMULATION"
    df["performance_ratio_used"] = pr_array
    return df


if __name__ == "__main__":
    # tiny smoke test
    n = 5
    demo = pd.DataFrame({
        "district_id": ["D42"] * n,
        "capacity_mw": [20.0] * n,
        "shortwave_radiation": [0, 200, 600, 900, 700],
        "clearsky_ghi_wm2": [0, 250, 650, 950, 750],
        "temperature_2m": [18, 22, 28, 33, 30],
        "daylight_flag": [0, 1, 1, 1, 1],
    })
    out = generate_proxy_production(demo)
    print(out[["shortwave_radiation", "pv_production_mw_proxy", "pv_production_source"]])
