# Enhancement changelog & audit findings

## Problems found in the original repo
1. `/forecast/*` returned the simulated *target* as `forecast_mw` (no model served) - and the dashboard plotted the target as the "forecast".
2. `/uncertainty` and dashboard page 4 used a rolling std of the target, not the fitted quantile models.
3. Script 11 (ENSTAB validation) needs `training_dataset_15min_real_reference.csv` and `data/validation/enstab_borj_cedria_real.csv`; both are git-ignored, so the ENSTAB validation could not be reproduced from the repo. Tables 1-2 of the report were TBD.
4. README test counts inconsistent (24 / 34 vs 54 collected); `xgboost` missing from requirements; Dockerfile lacked `libgomp1` (LightGBM would fail to import).

## Added / changed
* `src/forecasting/real_pv.py`, `scripts/12_real_benchmark.py` - measured-data benchmark (7 horizons, 5 models, conformal intervals, ablation, SHAP, figures, saved models)
* `src/forecasting/real_service.py` - live inference; `src/forecasting/national_hybrid.py` - ENSTAB-calibrated national forecast + transferred intervals
* API v0.4: honest `/forecast/*` (+`reference_mw`), `/uncertainty`, new `/real/benchmark|forecast|predict`
* Dashboard: pages 11 (real validation) and 12 (live replay); pages 1-5 show forecast vs reference
* Tests: +22 (leakage, conformal, artifacts, API, hybrid) -> 76 passed
* Deployment: Dockerfile fix, docker-compose, render.yaml, Streamlit config, GitHub Actions CI, Makefile, docs/DEPLOYMENT.md, docs/DEMO_SCRIPT.md
* Outputs: `data/validation/real_*.csv`, `real_test_forecasts.csv.gz`, `configs/calibration_enstab.json`, `reports/REAL_FORECAST_BENCHMARK.md`, `reports/figures/`, `reports/report_tables.tex`
