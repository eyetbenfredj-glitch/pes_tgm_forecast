# MODEL_VALIDATION_REPORT.md

**ALL numbers in this report were computed against the labelled DEMO/PROXY production target (`pv_production_mw_proxy`, `pv_production_source=DEMO_PROXY_PHYSICS_SIMULATION`). They demonstrate the pipeline is complete and correct; they are NOT a measurement of real-world forecasting skill, because no real STEG production data exists to validate against (see docs/REAL_PRODUCTION_DATA_SOURCES.md).**

- Dataset: `data/processed/training_dataset_15min_sample.csv`, 47,850 rows, 50 districts, 2026-06-25 00:00:00 -> 2026-07-04 23:00:00
- Chronological split: train <= 2026-06-30 23:24:00, val <= 2026-07-02 23:12:00, test <= 2026-07-04 23:00:00 (fractions 60/20/20 of the date range, no random shuffling)

## Model comparison (test set, national-level metrics)

|                      |    n |   MAE_MW |   RMSE_MW |   nMAE_pct_of_capacity |   nRMSE_pct_of_capacity |   bias_MW |   sMAPE_pct |
|:---------------------|-----:|---------:|----------:|-----------------------:|------------------------:|----------:|------------:|
| persistence          | 9600 |    0.294 |     0.63  |                  2.806 |                   4.57  |    -0     |      24.172 |
| seasonal_persistence | 9600 |    0.431 |     0.973 |                  4.698 |                   8.828 |    -0.02  |      28.938 |
| physics_baseline     | 9600 |    0.201 |     0.429 |                  1.92  |                   3.139 |    -0.016 |      20.851 |
| lightgbm             | 9600 |    0.221 |     0.454 |                  2.382 |                   4.101 |     0.033 |      61.455 |

## Uncertainty (quantile LightGBM + split-conformal)

- Quantile ordering (P10<=P25<=P50<=P75<=P90) respected on every test row: **True**
- 80% interval (P10-P90) empirical coverage: **84.5%** (nominal 80%)
- Mean interval width: **0.68 MW**
- Pinball losses: {'pinball_0.1': 0.052980386627624367, 'pinball_0.25': 0.08688663552050924, 'pinball_0.5': 0.10807794713396864, 'pinball_0.75': 0.08915156845316062, 'pinball_0.9': 0.051028644256255994}

## Hierarchical reconciliation

- Bottom-up reconciliation, sum(district)==governorate and sum(governorate)==national verified exactly: {'governorate_consistent': True, 'national_consistent': True, 'max_governorate_diff': 0.0, 'max_national_diff': 2.842170943040401e-14}

## Residual correction (LightGBM raw forecast, rolling bias corrector)

```json
{
  "raw": {
    "n": 4800,
    "MAE_MW": 0.22079210621615572,
    "RMSE_MW": 0.4536383494033146,
    "nMAE_pct_of_capacity": 2.3522241237500783,
    "nRMSE_pct_of_capacity": 4.013775152289257,
    "bias_MW": 0.030954616326390495,
    "sMAPE_pct": 61.87752372074389
  },
  "corrected": {
    "n": 4800,
    "MAE_MW": 0.21634506696393951,
    "RMSE_MW": 0.4428494115652875,
    "nMAE_pct_of_capacity": 2.2402768301645457,
    "nRMSE_pct_of_capacity": 3.6913748783517777,
    "bias_MW": 0.016464013563681296,
    "sMAPE_pct": 49.710140672032914
  }
}
```

## Physical validation of the target series

{'check_bounds_ok': 1.0, 'check_night_zero_ok': 1.0, 'check_no_jump_ok': 1.0, 'check_daylight_consistency_ok': 1.0, 'pct_flagged_anomalous': 0.0}

## Limitations

- Results are against a physics-derived proxy target - a model trained on it partly re-learns the physics equation used to generate it, so absolute error magnitudes are not informative about real-world skill; only structural correctness (ordering, coverage, reconciliation, correction direction) should be read from this report.
- Only the intra-day (15-min) horizon was backtested in this run; J+1/J+2/J+3 horizons are supported by `src/forecasting/backtesting.py::shift_to_horizon` but were not separately retrained/backtested here (next iteration).
- ML baseline excludes production lag features to stay honest about what would be available operationally at longer horizons; intra-day performance would improve with lag features once real production data justifies using them.
