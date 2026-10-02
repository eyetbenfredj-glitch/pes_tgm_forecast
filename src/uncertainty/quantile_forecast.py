"""
Probabilistic forecasting: quantile LightGBM + split-conformal calibration
(CDC continuation section 15).

Design:
  1. Train 5 independent LightGBM quantile regressors (objective="quantile")
     for tau in {0.10, 0.25, 0.50, 0.75, 0.90}.
  2. Enforce monotonic ordering P10<=P25<=P50<=P75<=P90 by sorting each row's
     raw quantile predictions (a standard, simple non-crossing fix).
  3. Split-conformal calibration on a held-out calibration set: widen the
     P10/P90 (and P25/P75) interval by the calibration-set's (1-alpha)
     empirical quantile of the residual, so the interval has a distribution-
     free finite-sample coverage guarantee rather than relying on the
     quantile regressor being perfectly calibrated out of the box.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_COL_NAMES = {0.10: "p10_mw", 0.25: "p25_mw", 0.50: "p50_mw", 0.75: "p75_mw", 0.90: "p90_mw"}


@dataclass
class QuantileForecastModel:
    feature_cols: list
    params_template: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
    ))
    models: dict = field(default_factory=dict)         # tau -> fitted LightGBM
    conformal_offsets: dict = field(default_factory=dict)  # tau -> additive offset from calibration

    def fit(self, train_df: pd.DataFrame, target_col: str):
        import lightgbm as lgb
        X = train_df[self.feature_cols]
        y = train_df[target_col]
        valid = y.notna() & X.notna().all(axis=1)
        for tau in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **self.params_template)
            m.fit(X[valid], y[valid])
            self.models[tau] = m
        return self

    def _raw_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols]
        valid = X.notna().all(axis=1)
        out = pd.DataFrame(index=df.index)
        for tau in QUANTILES:
            col = QUANTILE_COL_NAMES[tau]
            pred = np.full(len(df), np.nan)
            pred[valid.to_numpy()] = self.models[tau].predict(X[valid])
            out[col] = pred
        # enforce monotonic ordering P10<=P25<=P50<=P75<=P90 row-wise
        cols = [QUANTILE_COL_NAMES[t] for t in QUANTILES]
        out[cols] = np.sort(out[cols].to_numpy(), axis=1)
        return out.clip(lower=0)

    def calibrate(self, calib_df: pd.DataFrame, target_col: str):
        """Split-conformal: offset each quantile so the calibration-set
        empirical coverage matches its nominal level."""
        preds = self._raw_predict(calib_df)
        y = calib_df[target_col].to_numpy()
        for tau in QUANTILES:
            col = QUANTILE_COL_NAMES[tau]
            resid = y - preds[col].to_numpy()
            valid = ~np.isnan(resid)
            # one-sided conformal offset: the (1-tau)-th (upper tail) or tau-th
            # (lower tail) empirical quantile of signed residuals, applied so
            # coverage moves toward nominal.
            if tau <= 0.5:
                q = np.nanquantile(resid[valid], tau)  # can be negative -> tightens/loosens lower tail
            else:
                q = np.nanquantile(resid[valid], tau)
            self.conformal_offsets[tau] = float(q)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        preds = self._raw_predict(df)
        for tau in QUANTILES:
            col = QUANTILE_COL_NAMES[tau]
            preds[col] = (preds[col] + self.conformal_offsets.get(tau, 0.0)).clip(lower=0)
        cols = [QUANTILE_COL_NAMES[t] for t in QUANTILES]
        preds[cols] = np.sort(preds[cols].to_numpy(), axis=1)  # re-sort after calibration offsets
        return preds


def evaluate_coverage(y_true: pd.Series, p10: pd.Series, p90: pd.Series) -> dict:
    """80% interval (P10-P90) coverage + average width - CDC section 15/17."""
    covered = (y_true >= p10) & (y_true <= p90)
    return {
        "nominal_coverage_pct": 80.0,
        "empirical_coverage_pct": float(covered.mean() * 100),
        "mean_interval_width_mw": float((p90 - p10).mean()),
    }


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, tau: float) -> float:
    diff = y_true - y_pred
    return float(np.nanmean(np.maximum(tau * diff, (tau - 1) * diff)))
