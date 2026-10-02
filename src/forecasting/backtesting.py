"""
Chronological (walk-forward) backtesting + multi-horizon evaluation metrics.
CDC continuation sections 12, 16, 17.

STRICT RULE: splits are always chronological (train < validation < test by
timestamp). No random shuffling is ever applied to time-indexed data here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ChronoSplit:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    def split(self, df: pd.DataFrame, ts_col: str = "timestamp"):
        ts = df[ts_col]
        train = df[ts <= self.train_end]
        val = df[(ts > self.train_end) & (ts <= self.val_end)]
        test = df[(ts > self.val_end) & (ts <= self.test_end)]
        return train, val, test


def make_chrono_split(df: pd.DataFrame, ts_col: str = "timestamp",
                       train_frac: float = 0.6, val_frac: float = 0.2) -> ChronoSplit:
    """Derives split boundaries from data range fractions (documented, not random)."""
    ts_sorted = df[ts_col].sort_values()
    t0, t1 = ts_sorted.iloc[0], ts_sorted.iloc[-1]
    span = (t1 - t0)
    train_end = t0 + span * train_frac
    val_end = t0 + span * (train_frac + val_frac)
    return ChronoSplit(train_end=train_end, val_end=val_end, test_end=t1)


# ---------------------------------------------------------------- metrics --

def mae(y_true, y_pred) -> float:
    return float(np.nanmean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def nmae(y_true, y_pred, normalizer) -> float:
    """Normalized MAE, e.g. by installed capacity or by mean(y_true)."""
    n = np.asarray(normalizer)
    return float(np.nanmean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) / np.where(n == 0, np.nan, n)))


def nrmse(y_true, y_pred, normalizer) -> float:
    n = np.asarray(normalizer)
    return float(np.sqrt(np.nanmean(((np.asarray(y_true) - np.asarray(y_pred)) / np.where(n == 0, np.nan, n)) ** 2)))


def bias(y_true, y_pred) -> float:
    return float(np.nanmean(np.asarray(y_pred) - np.asarray(y_true)))


def smape(y_true, y_pred, eps: float = 1e-6) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.nanmean(np.abs(y_pred - y_true) / denom) * 100)


def evaluate(y_true: pd.Series, y_pred: pd.Series, capacity: pd.Series) -> dict:
    return {
        "n": int(y_true.notna().sum()),
        "MAE_MW": mae(y_true, y_pred),
        "RMSE_MW": rmse(y_true, y_pred),
        "nMAE_pct_of_capacity": nmae(y_true, y_pred, capacity) * 100,
        "nRMSE_pct_of_capacity": nrmse(y_true, y_pred, capacity) * 100,
        "bias_MW": bias(y_true, y_pred),
        "sMAPE_pct": smape(y_true, y_pred),
    }


def evaluate_by_group(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                       capacity_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        m = evaluate(g[y_true_col], g[y_pred_col], g[capacity_col])
        row = dict(zip(group_cols, keys))
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------- multi-horizon --

HORIZON_STEPS_15MIN = {
    "15min": 1, "1h": 4, "6h": 24, "12h": 48, "24h": 96,
    "J+1": 96, "J+2": 192, "J+3": 288,
}


def shift_to_horizon(df: pd.DataFrame, target_col: str, horizon_steps: int,
                      group_col: str = "district_id") -> pd.Series:
    """
    Builds the "target at horizon h" column: for a row at issue time t, the
    label is the actual value at t+h. Used to structure a supervised
    multi-horizon training set explicitly (CDC: never fake J+2/J+3 by copying
    J+1 - each horizon gets its own explicit label column here).
    """
    df = df.sort_values([group_col, "timestamp"])
    return df.groupby(group_col)[target_col].shift(-horizon_steps)
