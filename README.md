# PESTGM 7.0 - Track 1 · Shams'Na
## National Intelligent Platform for Forecasting Rooftop Solar Production (Tunisia)

[![CI](https://github.com/amalbenghnia/pestgm-pv-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/amalbenghnia/pestgm-pv-forecast/actions)

**Live demo:** dashboard `<STREAMLIT_URL>` · API docs `<RENDER_URL>/docs`  *(fill in after deploying - see `docs/DEPLOYMENT.md`)*

> **Tests:** `pytest tests -q` -> 76 passed, 5 skipped. **Real-data results:** `reports/REAL_FORECAST_BENCHMARK.md`.

## 1. What is real and what is not  (read first)

| Layer | Class | Note |
|---|---|---|
| Prosol fleet capacity (50 districts, 3 snapshots) | REAL | official |
| **ENSTAB Borj Cedria production + irradiance + met** | **MEASURED_REAL** | 2022-02 -> 2024-05, 5 min, ~3 kWc *(capacity estimated)* - the only measured data |
| National district-level production target | SIMULATED (`DEMO_PROXY_PHYSICS_SIMULATION`) | no public measured national series exists |
| Weather in the shipped national sample | `SYNTHETIC_CLEARSKY` | Open-Meteo/PVGIS blocked in the build sandbox; run `scripts/02` locally for real weather |

`DATA_MODE=real` refuses to run on synthetic data (`src/ingestion/data_mode.py`).

**Therefore:** accuracy claims are made **only** on the measured ENSTAB system. The national platform applies the *same validated method* to the fleet, and is labelled as such everywhere.

## 2. Measured results (ENSTAB, chronological hold-out Dec 2023 -> May 2024)
nMAE = daytime MAE as % of Pnom (3 kWc). Train 2022-03 -> 2023-05, validation (model selection + conformal calibration) 2023-06 -> 2023-11, test never touched.

| model                               |   15min |    1h |    3h |    6h |   J+1 |   J+2 |   J+3 |
|:------------------------------------|--------:|------:|------:|------:|------:|------:|------:|
| Climatology (month x hour)          |    9.75 |  9.75 |  9.75 |  9.75 |  9.75 |  9.75 |  9.75 |
| LightGBM                            |    3.36 |  5.91 |  7.72 |  8.45 |  8.98 |  9.81 |  9.92 |
| Persistence                         |    4.14 | 10.64 | 21.96 | 28.14 | 11.27 | 11.2  | 11.78 |
| Smart persistence (clear-sky index) |    3.42 |  7.52 | 14.83 | 18.27 | 12.54 | 12.53 | 12.96 |
| XGBoost                             |    3.37 |  5.99 |  7.8  |  8.4  |  8.88 |  9.72 |  9.93 |
| LightGBM P10-P90 coverage (%)       |   79    | 81    | 82    | 79    | 80    | 82    | 85    |

* 15-min: ML ≈ smart persistence (both strong); 1 h -> 6 h: ML clearly better (**21 % (1 h) to 54 % (6 h) lower MAE** than smart persistence).
* J+1: ML is **~20 % better than persistence** and ~8 % better than climatology.
* **J+2/J+3: ML ≈ climatology.** With no weather forecast there is no signal beyond ~1 day - this quantifies the value of adding NWP irradiance (next step).
* Conformal P10-P90 intervals cover **79-85 %** vs 80 % nominal (raw quantile models only 69-76 %).
* Physics transfer: the uncalibrated national config (PR = 0.80) shows a **-133 W bias** on real data; calibrating on ENSTAB brings it to **+9 W** (nMAE 5.5 % -> 4.2 %, same-time diagnostic with measured GHI).

## 3. Quick start
```bash
pip install -r requirements.txt
pytest tests -q                              # 76 tests
python scripts/12_real_benchmark.py          # ~80 s: regenerates every measured result, figures, models
uvicorn src.api.main:app --port 8000         # http://localhost:8000/docs
streamlit run src/dashboard/app.py           # 12 pages + Tunisia map
docker compose up --build                    # both, one command
```
Key new endpoints: `GET /real/benchmark`, `GET /real/forecast`, `GET /real/predict?issue_time=2024-04-10T10:00&horizon=J%2B1` (live model inference), plus the original `/forecast/*`, `/uncertainty`, `/export`...

## 4. Architecture
```
src/ingestion/     fleet, capacity interpolation, weather, PVGIS, temporal disaggregation, DATA_MODE guard
src/forecasting/   real_pv.py (measured-data pipeline)  real_service.py (live inference)  national_hybrid.py  backtesting.py  multi_horizon.py
src/physics/       pv_model.py            src/uncertainty/  quantile + conformal
src/aggregation/   bottom-up district -> governorate -> national
src/api/ main.py   FastAPI (17 endpoints)      src/dashboard/ app.py   Streamlit (12 pages + map)
scripts/           00-11 national pipeline | 12_real_benchmark.py (measured validation)
configs/calibration_enstab.json   physics parameters fitted on measured data
models/registry/real_enstab/      7 trained horizon models (LightGBM mean/P10/P90 + conformal qhat)
```
Method: clear-sky power `pcs` (pvlib) x ML-predicted clear-sky index `pk = P/pcs`; direct per-horizon models; features only from data available at issue time (`tests/test_real_pv.py::test_no_future_leakage` corrupts all future data and checks features are unchanged); forecasts forced to `0 <= P <= Pnom`, night = 0.

## 5. Limitations (stated up front)
1. Only **one** measured site (~3 kWc, capacity estimated, Borj Cedria). Test period is winter/spring only.
2. No NWP forecast for ENSTAB: day-ahead models are a floor (history + astronomy only).
3. Calibration implies effective PR ≈ 0.93 at the estimated 3 kWc - either nameplate < 3 kWc or the irradiance sensor is plane-of-array. Confirm both before trusting absolute national MW.
4. National forecast intervals are transferred from one site (conservative for aggregates).
5. Fleet coordinates are representative points, not administrative polygons.
