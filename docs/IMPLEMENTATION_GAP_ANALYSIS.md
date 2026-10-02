# IMPLEMENTATION_GAP_ANALYSIS.md

Audit performed before this iteration's coding work, against the existing
repository state (previous session: 24/40 DoD items, 26 PASS/8 PARTIAL/6
BLOCKED per `FINAL_CDC_COMPLIANCE_MATRIX.md`).

| Area | Status before this session | Gap / what must change |
|---|---|---|
| Real weather ingestion | PARTIAL/BLOCKED | Re-verified this session: `archive-api.open-meteo.com` and `re.jrc.ec.europa.eu` both return HTTP 403 from this sandbox (direct curl test). No change possible without an unrestricted network. Action: document precisely (done), make the code runnable elsewhere (already true), do not re-attempt to fake it. |
| PVGIS integration | BLOCKED (evaluated only, no code) | Build `src/ingestion/pvgis_provider.py` as real, runnable client with retry/caching/provenance, even though unreachable here. |
| Real PV production search | PARTIAL (STEG/ANME/data.gov.tn/IRENA/JODI checked) | Extend to Zenodo, OpenAIRE, Kaggle, academic literature. |
| Multi-horizon (J+1/J+2/J+3) | PARTIAL (`shift_to_horizon` utility existed, not used to actually train/backtest separate horizon models) | Implement Option A: one model per horizon, actually fit + backtest each, per CDC Phase 5/6. |
| Uncertainty per horizon | PARTIAL (only intra-day) | Extend quantile+conformal to each horizon. |
| Residual correction per horizon | PARTIAL (single intra-day experiment) | Extend to each horizon; keep no-future-actuals guarantee. |
| Continuous learning | PARTIAL (`.fit()/.correct()` mechanism existed, no standing script) | Build `scripts/06_update_forecast_model.py` with explicit workflow + versioning. |
| Drift/anomaly monitoring | BLOCKED (nothing implemented) | Build `src/monitoring/` (weather/production/forecast-error drift, anomaly flags) + `GET /anomalies`. |
| Explainability (SHAP) | BLOCKED (dependency installed, unused) | Wire SHAP to the LightGBM models, save global + local explanations under `reports/explainability/`. |
| API completion | PARTIAL (6/14 endpoints) | Add `/forecast/{horizon}`, `POST /forecast`, `/uncertainty`, `/errors`, `/anomalies`, CSV/JSON export. |
| Grid/load export | PARTIAL (schema only) | Add explicit CSV/JSON export endpoints/functions. |
| Dashboard | BLOCKED | Build Streamlit app, 10 pages as specified. |
| Tunisia map | BLOCKED | Build inside the dashboard using the existing 50 representative coordinates. |
| Physical validation of forecasts | PARTIAL (only applied to the target series) | Apply `run_all_checks` to model forecasts (raw, corrected, quantile) too. |
| Tests | 24/24 passing, narrower scope | Extend for every new module below; keep all 24 existing passing. |
| Reports | `MODEL_VALIDATION_REPORT.md` (DEMO only, intra-day only) | Extend per-horizon, add explicit REAL/DEMO section headers (REAL section will state "no real data" rather than being silently omitted). |
| Technical report | Not created | `reports/PESTGM_TECHNICAL_REPORT.md`. |

**Explicit non-goal restated:** no component below is rebuilt or replaced
unless integration requires it. Fleet loader, capacity time series, solar
geometry, synthetic weather provider, physics model, hierarchical
reconciliation (bottom-up) and DATA_MODE enforcement are already correct and
tested — reused as-is.

This document is superseded by the final, post-implementation state of
`FINAL_CDC_COMPLIANCE_MATRIX.md`, which reflects what was ACTUALLY done by
the end of this session (this file records intent/plan, not completion).
