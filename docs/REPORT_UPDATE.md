# What to change in the technical report (paste-ready)

All numbers: `reports/REAL_FORECAST_BENCHMARK.md` (auto-generated). LaTeX tables: `reports/report_tables.tex`. Figures: `reports/figures/fig1..fig5.png`.

## Section 9.3 / Table 1 - replace TBD (J+1, ENSTAB test Dec 2023-May 2024)
| Model | MAE (W) | RMSE (W) | nMAE (%) | Bias (W) |
|---|---|---|---|---|
| Persistence (same time, last available day) | 338.1 | 528.3 | 11.27 | -7.4 |
| Smart persistence (clear-sky index) | 376.3 | 496.5 | 12.54 | -84.4 |
| Physics reference, clear-sky (no cloud info) | 829.0 | 1024.9 | 27.63 | +816.7 |
| Physics reference, calibrated, measured GHI (same-time diagnostic, not a forecast) | 126.4 | 187.6 | 4.21 | +9.1 |
| LightGBM | 269.5 | 379.2 | 8.98 | +8.5 |
| XGBoost | 266.3 | 377.3 | 8.88 | +1.3 |

## Section 11 / Table 2 - LightGBM per horizon (test set)
| Horizon | MAE (W) | RMSE (W) | nMAE (%) | P10-P90 coverage (%) |
|---|---|---|---|---|
| Intraday (15 min) | 100.9 | 176.7 | 3.36 | 78.6 |
| 1 h | 177.2 | 274.1 | 5.91 | 81.3 |
| 3 h | 231.6 | 331.9 | 7.72 | 82.4 |
| 6 h | 253.5 | 361.7 | 8.45 | 79.3 |
| J+1 | 269.5 | 379.2 | 8.98 | 80.0 |
| J+2 | 294.4 | 404.5 | 9.81 | 82.0 |
| J+3 | 297.5 | 406.1 | 9.92 | 85.4 |

## New paragraph for Section 11 (Results)
"All results below are computed against measured production from the ENSTAB Borj Cedria PV system on a chronological hold-out (December 2023 - May 2024) that was never used for training, model selection or interval calibration. At 15-minute lead time the hybrid model reaches an nMAE of 3.4 % of nominal power, comparable to smart persistence; from 1 h to 6 h it reduces MAE by 21-54 % relative to clear-sky-index persistence. At J+1 it is 20 % better than persistence and 8 % better than climatology. Beyond J+1 the models perform at climatological level (nMAE 9.8-9.9 %), which is expected because no numerical weather forecast is available for this site; this quantifies the benefit that NWP irradiance forecasts are expected to bring. Split-conformal P10-P90 intervals cover 79-85 % of observations against an 80 % nominal level, whereas the raw quantile models cover only 69-76 %."

## New paragraph for Section 13 (Limitations) - add
* "Measured validation is limited to one ~3 kWc system, with estimated nameplate capacity, over winter-spring."
* "The calibrated physical model implies an effective performance ratio of ~0.93 at the estimated capacity, indicating either a lower nameplate capacity or a plane-of-array irradiance sensor; absolute national MW values should be confirmed with these two facts."
* "National-level intervals are transferred from a single site and are therefore conservative."

## Section 3.3 - add one sentence
"The ENSTAB dataset is used (i) as external validation and (ii) to fit only two physical parameters (effective efficiency and temperature coefficient) of the physics model, on the training split."

## Figures to include
fig1 (nMAE vs horizon), fig2 (interval reliability), fig3 (clear/cloudy day examples), fig4 (SHAP), fig5 (ablation).
