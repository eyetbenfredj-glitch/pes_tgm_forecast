"""
Real-data forecasting library (MEASURED_REAL: ENSTAB Borj Cedria rooftop PV).

Everything here works on *measured* production, so the numbers it produces are
legitimate measured-performance numbers (unlike the DEMO/PROXY national target).

Design (hybrid physical + ML):
  * pcs   = clear-sky power reference  (pvlib clear-sky GHI x site constant)
  * pk    = p / pcs                    (clear-sky index of POWER, the ML target)
  * P_hat = clip(pk_hat * pcs, 0, Pnom), forced to 0 at night
Forecasts are DIRECT and per-horizon. Features are computed ONLY from
information available at the forecast issue time + deterministic "known future"
quantities (sun position, clear-sky). No future measurement ever leaks in
(see tests/test_real_pv.py).

Horizons
  intraday : 15min, 1h, 3h, 6h  -> issued every 15 min, lead = horizon
  day-ahead: J+1, J+2, J+3      -> issued at ISSUE_HOUR (10:00 local) of day D
                                   for every 15-min step of day D+k
NOTE: no NWP weather forecast is available for ENSTAB, so day-ahead models use
only measured history + astronomy. They are therefore a *floor*: adding NWP
irradiance forecasts is the identified next step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

SITE = dict(lat=36.72, lon=10.43, alt=30.0, tz="Africa/Tunis")
PNOM_W = 3000.0            # ESTIMATED nameplate (see enstab_validator.py), not measured
STEPS_PER_DAY = 96
ISSUE_HOUR = 10
DAY_PCS_MIN_W = 20.0       # a timestep is "daytime" if clear-sky power > this

HORIZONS = {
    "15min": ("intraday", 1), "1h": ("intraday", 4), "3h": ("intraday", 12), "6h": ("intraday", 24),
    "J+1": ("dayahead", 1), "J+2": ("dayahead", 2), "J+3": ("dayahead", 3),
}
SPLITS = {   # by TARGET time. Chronological, no shuffling.
    "train": ("2022-03-01", "2023-05-31"),
    "val": ("2023-06-01", "2023-11-30"),
    "test": ("2023-12-01", "2024-05-31"),
}
HIST = ["pk", "pk_ffill", "age_h", "pk_m4", "pk_m12", "pk_sd4", "pk_d4", "kt_meas", "kt_meas_m4",
        "pk_day1", "pk_day3", "pk_day7", "pk_day30", "pk_lag_same"]
MET = ["t_amb", "rh", "wind", "pres", "dpres3h"]
KNOWN = ["ghi_cs", "elev", "azim", "hsin", "hcos", "dsin", "dcos", "lead_h"]
FEATURE_SETS = {
    "full": HIST + MET + KNOWN,
    "no_history": KNOWN,                       # astronomy only = "climatology"
    "no_met": HIST + KNOWN,                    # without local met sensors
}


# --------------------------------------------------------------------------- data
def load_enstab(path: str | Path = "data/external/enstab_borj_cedria.csv") -> pd.DataFrame:
    """Load raw 5-min ENSTAB data -> 15-min means (W, W/m2, degC...), local naive index."""
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    raw = raw.rename(columns={"Power": "p", "Irradiation": "ghi", "Temperature": "t_amb",
                              "Cell_temperature": "t_cell", "Humidity": "rh",
                              "Wind_speed": "wind", "Pressure": "pres"})
    df = raw.resample("15min").mean()
    return add_solar(df)


def add_solar(df: pd.DataFrame, site: dict = SITE) -> pd.DataFrame:
    """Add pvlib sun position + clear-sky GHI at interval midpoints and pk target."""
    df = df.copy()
    loc = pvlib.location.Location(site["lat"], site["lon"], tz=site["tz"], altitude=site["alt"])
    mid = (df.index + pd.Timedelta(minutes=7.5)).tz_localize(site["tz"])
    sp = loc.get_solarposition(mid)
    cs = loc.get_clearsky(mid, model="ineichen", linke_turbidity=3.0, solar_position=sp)
    df["elev"] = sp["apparent_elevation"].values
    df["azim"] = sp["azimuth"].values
    df["ghi_cs"] = cs["ghi"].values
    return df


def fit_clearsky_power_constant(df_train: pd.DataFrame) -> float:
    """c such that pcs = c * ghi_cs: 97th percentile of P/GHI_cs on clear train samples."""
    m = (df_train["ghi_cs"] > 400) & (df_train["ghi"] / df_train["ghi_cs"] > 0.85)
    ratio = (df_train.loc[m, "p"] / df_train.loc[m, "ghi_cs"]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.percentile(ratio, 97))


def add_targets(df: pd.DataFrame, c: float) -> pd.DataFrame:
    df = df.copy()
    df["pcs"] = (c * df["ghi_cs"]).clip(lower=0)
    day = df["pcs"] > DAY_PCS_MIN_W
    df["day"] = day
    df["pk"] = (df["p"] / df["pcs"]).where(day).clip(0, 1.5)
    df["kt_meas"] = (df["ghi"] / df["ghi_cs"]).where(df["elev"] > 5).clip(0, 1.5)
    return df


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features indexed by the time they are OBSERVED (use only past/present at that time)."""
    pk = df["pk"]
    f = pd.DataFrame(index=df.index)
    f["pk"] = pk
    f["pk_ffill"] = pk.ffill(limit=STEPS_PER_DAY)
    valid = pk.notna()
    last_valid = pd.Series(np.where(valid, np.arange(len(df)), np.nan), index=df.index).ffill()
    f["age_h"] = (np.arange(len(df)) - last_valid) * 0.25
    f["pk_m4"] = pk.rolling(4, min_periods=1).mean()
    f["pk_m12"] = pk.rolling(12, min_periods=1).mean()
    f["pk_sd4"] = pk.rolling(4, min_periods=2).std()
    f["pk_d4"] = pk - pk.shift(4)
    f["kt_meas"] = df["kt_meas"]
    f["kt_meas_m4"] = df["kt_meas"].rolling(4, min_periods=1).mean()
    for name, win in [("pk_day1", 96), ("pk_day3", 288), ("pk_day7", 672), ("pk_day30", 2880)]:
        f[name] = pk.rolling(win, min_periods=max(8, win // 12)).mean()
    for c in ["t_amb", "rh", "wind", "pres"]:
        f[c] = df[c]
    f["dpres3h"] = df["pres"] - df["pres"].shift(12)
    return f


def _known(df: pd.DataFrame, t_idx: np.ndarray) -> pd.DataFrame:
    ts = df.index[t_idx]
    mid = ts + pd.Timedelta(minutes=7.5)
    hod = mid.hour + mid.minute / 60
    doy = mid.dayofyear
    return pd.DataFrame({
        "ghi_cs": df["ghi_cs"].values[t_idx], "elev": df["elev"].values[t_idx],
        "azim": df["azim"].values[t_idx],
        "hsin": np.sin(2 * np.pi * hod / 24), "hcos": np.cos(2 * np.pi * hod / 24),
        "dsin": np.sin(2 * np.pi * doy / 365.25), "dcos": np.cos(2 * np.pi * doy / 365.25),
    })


def build_dataset(df: pd.DataFrame, feats: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Return one row per (issue, target) pair: features + y(pk) + baselines inputs.

    Leak-free by construction: every HIST/MET value is read at index s_idx (the
    issue time) and `pk_lag_same` at t - 96*ceil(lead/96) <= issue time.
    """
    kind, v = HORIZONS[horizon]
    n = len(df)
    if kind == "intraday":
        t_idx = np.arange(v, n)
        s_idx = t_idx - v
    else:
        hours = df.index.hour.values
        minutes = df.index.minute.values
        issue = np.where((hours == ISSUE_HOUR) & (minutes == 0))[0]
        s_list, t_list = [], []
        for i in issue:
            start = i - ISSUE_HOUR * 4 + STEPS_PER_DAY * v
            if start + STEPS_PER_DAY > n:
                continue
            tt = np.arange(start, start + STEPS_PER_DAY)
            t_list.append(tt)
            s_list.append(np.full(STEPS_PER_DAY, i))
        t_idx, s_idx = np.concatenate(t_list), np.concatenate(s_list)
    lead = t_idx - s_idx
    lag_days = np.ceil(lead / STEPS_PER_DAY).astype(int)
    lag_idx = t_idx - STEPS_PER_DAY * np.maximum(lag_days, 1)
    ok = (s_idx >= 0) & (lag_idx >= 0)
    t_idx, s_idx, lead, lag_idx = t_idx[ok], s_idx[ok], lead[ok], lag_idx[ok]

    X = feats.iloc[s_idx].reset_index(drop=True)
    X["pk_lag_same"] = df["pk"].values[lag_idx]
    X = pd.concat([X, _known(df, t_idx)], axis=1)
    X["lead_h"] = lead * 0.25
    X["t"] = df.index[t_idx]
    X["issue"] = df.index[s_idx]
    X["pcs"] = df["pcs"].values[t_idx]
    X["p"] = df["p"].values[t_idx]
    X["day"] = df["day"].values[t_idx]
    X["y"] = df["pk"].values[t_idx]
    X["p_persist"] = np.where(kind == "intraday", df["p"].values[s_idx], df["p"].values[lag_idx])
    key = "pk_ffill" if kind == "intraday" else "pk_day1"
    kt_issue = feats[key].values[s_idx]
    kt_issue = np.where(np.isnan(kt_issue), feats["pk_ffill"].values[s_idx], kt_issue)
    X["kt_issue"] = np.clip(kt_issue, 0, 1.2)
    return X


def split_mask(t: pd.Series, name: str) -> np.ndarray:
    a, b = SPLITS[name]
    return ((t >= pd.Timestamp(a)) & (t < pd.Timestamp(b) + pd.Timedelta(days=1))).values


# --------------------------------------------------------------------------- models
def apply_constraints(p_hat: np.ndarray, pcs: np.ndarray, pnom: float = PNOM_W) -> np.ndarray:
    """Eq. (2)-(3) of the report: 0 <= P <= Pnom and P = 0 at night."""
    out = np.clip(p_hat, 0, pnom)
    return np.where(pcs > DAY_PCS_MIN_W, out, 0.0)


def climatology_table(train: pd.DataFrame) -> pd.Series:
    d = train[train["day"]].copy()
    d["m"] = d["t"].dt.month
    d["h"] = d["t"].dt.hour
    return d.groupby(["m", "h"])["y"].mean()


def climatology_pred(d: pd.DataFrame, table: pd.Series) -> np.ndarray:
    key = list(zip(d["t"].dt.month, d["t"].dt.hour))
    fallback = float(table.mean())
    return np.array([table.get(k, fallback) for k in key]) * d["pcs"].values


LGB_PARAMS = dict(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=60,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                  random_state=42, verbose=-1, n_jobs=4)


LGB_PRESETS = {
    "A_default": {},
    "B_regularised": dict(num_leaves=15, min_child_samples=200, n_estimators=180),
    "C_strong": dict(num_leaves=7, min_child_samples=500, n_estimators=140),
}


def select_preset(Xtr, ytr, wtr, Xva, yva, wva) -> str:
    """Pick the regularisation preset with the lowest weighted VALIDATION MAE (never the test set)."""
    best, best_mae = None, np.inf
    for name, extra in LGB_PRESETS.items():
        m = fit_lgbm(Xtr, ytr, wtr, extra=extra)
        mae = float(np.average(np.abs(m.predict(Xva) - yva), weights=wva))
        if mae < best_mae:
            best, best_mae = name, mae
    return best


def fit_lgbm(X, y, w, objective="regression", alpha=None, extra=None):
    import lightgbm as lgb
    p = dict(LGB_PARAMS, objective=objective, **(extra or {}))
    if alpha is not None:
        p["alpha"] = alpha
    m = lgb.LGBMRegressor(**p)
    m.fit(X, y, sample_weight=w)
    return m


def fit_xgb(X, y, w):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                         random_state=42, n_jobs=4, tree_method="hist")
    m.fit(X, y, sample_weight=w)
    return m


def conformal_qhat(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float = 0.2) -> float:
    """Split-conformal CQR (Romano et al. 2019), in clear-sky-index units."""
    s = np.maximum(lo - y, y - hi)
    n = len(s)
    q = min(1.0, (1 - alpha) * (1 + 1 / n))
    return float(np.quantile(s, q))


# --------------------------------------------------------------------------- metrics
def metrics(p: np.ndarray, p_hat: np.ndarray, frac_day: float = 1.0, pnom: float = PNOM_W) -> dict:
    """Daytime metrics + 24h-equivalent nMAE (night zeros included analytically)."""
    e = p_hat - p
    mae = float(np.mean(np.abs(e)))
    ss_res, ss_tot = float(np.sum(e ** 2)), float(np.sum((p - p.mean()) ** 2))
    return dict(MAE_W=mae, RMSE_W=float(np.sqrt(np.mean(e ** 2))),
                nMAE_pct=100 * mae / pnom, nRMSE_pct=100 * float(np.sqrt(np.mean(e ** 2))) / pnom,
                nMAE24_pct=100 * mae * frac_day / pnom, bias_W=float(np.mean(e)),
                R2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan, n=int(len(p)))


# --------------------------------------------------------------------------- physics
def fit_calibrated_physics(df_train: pd.DataFrame) -> dict:
    """OLS on measured data: P/G = A * (1 + gamma*(Tcell-25)) with G>100 W/m2 (train only)."""
    d = df_train[(df_train["ghi"] > 100)].copy()
    y = d["p"] / d["ghi"]
    x = d["t_cell"] - 25.0
    slope, intercept = np.polyfit(x, y, 1)
    return dict(A_w_per_wm2=float(intercept), gamma_per_c=float(slope / intercept),
                pnom_w=PNOM_W, A_norm=float(intercept / PNOM_W),
                note="Fitted on ENSTAB TRAIN split only (measured GHI + measured cell temperature).")


def physics_power(ghi, t_cell, cal: dict):
    return np.clip(cal["A_w_per_wm2"] * ghi * (1 + cal["gamma_per_c"] * (t_cell - 25.0)), 0, PNOM_W)


def national_config_physics(ghi, t_amb, pr=0.80, gamma=-0.004, noct=45.0, pnom=PNOM_W):
    """The *uncalibrated* national reference config (configs/config.yaml) applied to a 3 kWc system."""
    t_cell = t_amb + (noct - 20.0) / 800.0 * ghi
    return np.clip(pnom * pr * ghi / 1000.0 * (1 + gamma * (t_cell - 25.0)), 0, pnom)


def save_json(obj: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
