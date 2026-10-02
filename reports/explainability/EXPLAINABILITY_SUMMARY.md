# Explainability summary (SHAP, LightGBM)

Model: LightGBMBaseline (intraday) | Trained on: DEMO_PROXY_PHYSICS_SIMULATION

## Global feature importance (mean |SHAP value|)

| feature             |   mean_abs_shap |
|:--------------------|----------------:|
| shortwave_radiation |      1.26781    |
| capacity_mw         |      1.00293    |
| direct_radiation    |      0.33917    |
| clearsky_ghi_wm2    |      0.0754745  |
| solar_elevation_deg |      0.0396601  |
| doy_sin             |      0.0238617  |
| diffuse_radiation   |      0.0216145  |
| temperature_2m      |      0.0201996  |
| cloud_cover         |      0.00935188 |
| hour_cos            |      0.00726016 |

## Local explanation example (district=TATAOUINE, timestamp=2026-06-27 15:00:00)

Base value: 1.6262 MW | Prediction: 2.0147 MW

| feature             |   feature_value |   shap_value |
|:--------------------|----------------:|-------------:|
| shortwave_radiation |     357.051     |   0.752267   |
| capacity_mw         |       7.58064   |  -0.561224   |
| direct_radiation    |     198.666     |   0.144552   |
| clearsky_ghi_wm2    |     790.701     |   0.074746   |
| solar_elevation_deg |      53.9509    |   0.0364564  |
| temperature_2m      |      32.9364    |  -0.0220138  |
| doy_sin             |       0.0794773 |  -0.0179381  |
| cloud_cover         |      53.9242    |  -0.00854377 |
| diffuse_radiation   |     212.451     |  -0.00606218 |
| solar_azimuth_deg   |     264.904     |  -0.00524927 |
