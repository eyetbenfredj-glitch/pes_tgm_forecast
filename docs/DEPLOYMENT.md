# Deployment (live prototype)

## A. Dashboard -> Streamlit Community Cloud (free, from GitHub)
1. Push the repo to GitHub (all `data/`, `models/registry/real_enstab/` and `configs/calibration_enstab.json` must be committed).
2. share.streamlit.io -> *New app* -> repo, branch `main`, **Main file path `src/dashboard/app.py`**.
3. The dashboard is self-contained (no API call needed). First load of page 12 takes ~10-15 s (loads the ENSTAB series and builds features once, then cached).

## B. API -> Render (free) or Hugging Face Spaces (Docker)
* Render: *New +* -> *Blueprint* -> select repo (`render.yaml` is included). Live docs at `https://<name>.onrender.com/docs`.
* Spaces: create a *Docker* Space, push the repo, set `app_port: 8000` in the Space README header.

## C. Everything locally
`docker compose up --build` -> API :8000, dashboard :8501.   or   `make api` / `make dashboard`.

## D. Make the national forecast use REAL weather (needs internet - blocked in the build sandbox)
```
DATA_MODE=demo python scripts/02_build_weather_dataset.py      # real Open-Meteo if reachable
DATA_MODE=demo python scripts/04_build_training_dataset.py --start 2026-06-25 --end 2026-07-05
```
Then commit the new `data/processed/*_sample.csv`. The hybrid forecast (`src/forecasting/national_hybrid.py`) picks the new irradiance up automatically.

## Checklist before the jury
- [ ] both URLs open in < 30 s and show the data-mode banner
- [ ] `pytest tests -q` green (CI badge)
- [ ] 2-3 min backup video recorded
