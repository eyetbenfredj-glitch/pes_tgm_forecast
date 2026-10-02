# 5-minute live demo script (judges)

Open the dashboard + `API /docs` in two tabs *before* the session (free tiers sleep - wake them up 5 min earlier).

| min | Show | Say |
|---|---|---|
| 0:00 | Banner on any page | "We never mix data classes: MEASURED_REAL / PHYSICAL_REFERENCE / SYNTHETIC_DEMO are always labelled." |
| 0:30 | **Tunisia map** -> **1. National overview** | "50 districts, official Prosol capacity, bottom-up reconciliation district -> governorate -> national (exact by construction)." |
| 1:30 | **3. District forecast**, **4. Uncertainty** | "Physics forecast calibrated on measured data + P10-P90 band." Be explicit: national target is simulated, the *method* is validated on real data. |
| 2:15 | **11. REAL validation (ENSTAB)** | "Chronological hold-out on real measured production: 15-min nMAE 3.4%, J+1 9.0%, 80% interval covers ~80%." Point at the J+2/J+3 = climatology result: "this quantifies why we need NWP irradiance next." |
| 3:30 | **12. Live forecast** | Pick a cloudy date, J+1. "The model is run now, features only from data before the issue time - then we reveal what was measured." |
| 4:30 | API `/docs` -> `/real/predict`, `/forecast/national`, `/export` | "Same engine through 14+ endpoints, CSV export for grid/load integration." |
| 5:00 | Limitations slide | Single real site, no NWP yet, capacity of ENSTAB estimated -> roadmap. |

Backup: record this flow as a 2-3 min screen video.
