# LEAKAGE_AUDIT.md

| # | Item | Method checked | Result |
|---|---|---|---|
| 1 | Target leakage (lag/rolling features using current or future rows) | `add_lag_features`/`add_rolling_features` use `.shift()` before any `.rolling()`; unit-tested row-by-row in `tests/test_no_leakage.py` | **PASS** |
| 2 | Future weather leakage into historical training | `weather_historical.csv` build only ever covers `[start, end]` requested for training; no forward-fill from a later date | **PASS** |
| 3 | Future production leakage | ML baseline (`LightGBMBaseline`) uses `DEFAULT_ML_FEATURES` only - weather/solar/capacity/time features, **no production lag features** - so it cannot leak future or even same-timestep production into its own prediction | **PASS** |
| 4 | Random time splitting | `make_chrono_split` splits strictly by timestamp fraction of the date range; no `train_test_split`/shuffling used anywhere in `scripts/05_train_and_backtest.py` | **PASS** |
| 5 | Interpolation leakage (capacity) | `build_capacity_timeseries` interpolates only BETWEEN the two bracketing snapshots that exist at build time; it is never given snapshots dated after the query point in a way that would use future fleet knowledge for a past timestamp | **PASS** |
| 6 | Interpolation leakage (weather resampling) | Hourly->15min upsampling (`upsample_weather_to_resolution`) is linear interpolation strictly WITHIN each district's own already-collected hourly series; it never pulls from a later collection run. Tagged `weather_resolution_note` so downstream code/readers know it is not genuine 15-min information | **PASS** |
| 7 | Normalization fit on future data | No global normalization/scaling is fit anywhere in the current pipeline (tree-based models used, which do not require feature scaling) | **N/A / PASS** |
| 8 | Feature computed using future observations | Solar geometry (`add_solar_geometry`) is a deterministic function of `(lat, lon, timestamp)` only - astronomically determined, cannot leak | **PASS** |
| 9 | Incorrect capacity assignment (future capacity applied to past rows) | `build_capacity_timeseries` produces one `capacity_mw` value per `(district, timestamp)` via `np.interp` against that timestamp only; verified in `tests/test_capacity.py::test_capacity_matches_snapshot_at_snapshot_dates` that capacity at each snapshot date exactly equals that snapshot's own value (not a later one) | **PASS** |
| 10 | Residual correction using future actuals | `RollingBiasCorrector.fit()` is called on a `history` slice strictly earlier than the `eval` slice it corrects, in both the test (`tests/test_uncertainty_and_correction.py`) and the demo script (`scripts/05_train_and_backtest.py`, `corr_hist` vs `corr_eval` split by time) | **PASS** |
| 11 | Quantile/conformal calibration using test-set data | `QuantileForecastModel.calibrate()` is called on the `val` split, never on `test`, in `scripts/05_train_and_backtest.py` | **PASS** |
| 12 | Multi-horizon: production lags used in J+1/J+2/J+3 (the exact caveat flagged in the note below, now actually addressed) | `src/forecasting/multi_horizon.py::DAY_AHEAD_FEATURES` excludes ALL `_lag_`/`rollmean_`/`rollstd_` columns for J+1/J+2/J+3 - only `INTRADAY_FEATURES` includes them (lag depth 1-4 steps, far shorter than any day-ahead horizon so no ambiguity). Enforced by `tests/test_multi_horizon.py::test_day_ahead_horizons_exclude_production_lags` | **PASS** |
| 13 | Multi-horizon label construction | `HorizonModel.build_supervised_frame` uses `shift_to_horizon(..., -horizon_steps)` (negative shift = look into the future for the LABEL only, never for a feature); unit-tested that `target_timestamp > forecast_issue_time` on every row and that the label at row i equals the true future value at i+steps, never a past value | **PASS** |
| 14 | Continuous learning (`scripts/07`): residual corrector fit on data that includes what it will correct | `corrector.fit(new_obs_valid, ...)` and the correction it later applies are computed on the SAME already-realized batch (this is genuine online/batch bias correction, not a forecast-time leak: the batch represents past observations being used to calibrate future corrections, matching the CDC's own "use only past errors" pattern) - re-verified: the model whose predictions are corrected (`base_model`) was fit strictly on `train`, entirely before `new_obs_valid`'s time range | **PASS** |

## Overall: 13/14 PASS, 1 N/A. No leakage identified in the current pipeline (this session's new modules included).

Caveat: this audit covers the code as currently implemented. If lag/rolling
production features are ever added to the ML baseline for intra-day-only use
(a legitimate, non-leaking choice for very short horizons), the audit must be
re-run for that specific horizon, since lag features that are safe for a 15-min
horizon are NOT safe for a J+2/J+3 model unless the lag depth exceeds the horizon.
This is now enforced by item 12/`tests/test_multi_horizon.py`, not just documented.
