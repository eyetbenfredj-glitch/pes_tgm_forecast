# REAL_PRODUCTION_DATA_SOURCES.md

Investigation conducted 2026-09-22 for PESTGM 7.0 Track 1. Goal: find a REAL
measured PV production time series (`timestamp | district | PV production MW`)
for Tunisia, at district, governorate, or national level.

## Method

Searched (web) + queried directly (where technically reachable): STEG,
ANME/Prosol, Tunisia's national open-data portal, IRENA, JODI, and known
research/simulation platforms (PVGIS, Renewables.ninja, NASA POWER). Also
attempted direct programmatic access to weather/irradiance APIs from this
build environment (see §4).

## A/B/C/D — Real MEASURED production (district / governorate / national / aggregated rooftop)

**Extended search pass (session 2, Zenodo/OpenAIRE/Kaggle/academic literature):** checked Zenodo, OpenAIRE, IEEE DataPort, Data in Brief, and 4TU.ResearchData for Tunisia-specific PV production datasets. Found: a Tataouine (Tunisia) **wind-speed** dataset (4TU, Melalkia & Berrezzek 2024) — not PV. Found multiple real PV production datasets for *other* countries (La Réunion rooftop PV, Myanmar residential PV-battery, pan-European PV capacity factors from Aarhus University/SARAH reanalysis) — none for Tunisia. **No Tunisia-specific measured PV production dataset was found on any of these platforms.**

| Source checked | What was found | Real measured production available? |
|---|---|---|
| **STEG** (steg.com.tn) — national utility | Public pages describe the Prosol program, net-metering (compteur bidirectionnel), tariffs, and one anecdotal 22 kWp demo installation at STEG HQ. National **total electricity** production (all sources, GWh/year, annual peak MW) is published in prose (e.g. 19,313 GWh in 2023, national peak 4,825 MW on 2023-07-20). **No PV-specific production time series, no district/governorate breakdown, no API or downloadable dataset found.** | **No** |
| **data.gov.tn** — Tunisia's national CKAN open-data portal | Portal exists and hosts government datasets (confirmed via the R package `tndata`, which wraps its CKAN API) across administrative/economic/statistical themes. No PV-production or energy-generation dataset was found in the sources checked. A direct CKAN `package_search` query for "photovoltaïque"/"énergie solaire" could not be executed from this environment (network-restricted; see §4) and was only checked via web search, which returned no matching dataset. | **Not confirmed absent, but not found — needs a direct CKAN query from an unrestricted environment** |
| **ANME** (Agence Nationale pour la Maîtrise de l'Énergie) — runs Prosol | Referenced as the source of the Prosol subsidy (1,500 DT/installation) in secondary sources. No public ANME open-data portal or production API was located. | **No** |
| **JODI** (Joint Organisations Data Initiative) | Tunisia participates (2014 workshop material found), but JODI covers aggregate monthly oil/gas/electricity balances, not sub-hourly PV generation. | **No** (wrong resolution/scope even if accessible) |
| **IRENA** statistics | Publishes Tunisia's **annual installed solar capacity** (e.g. 506 MW in 2023) and **renewable share of TFEC** — capacity and energy-balance statistics, not a production time series. | **No** |
| Kaggle / research datasets | No Tunisia-specific rooftop-PV production dataset located in this search pass. | **No** (not exhaustively ruled out — worth a dedicated Kaggle/Zenodo/OpenAIRE search pass) |

**Conclusion: no real, measured, timestamped PV production dataset for Tunisia (district, governorate, or national) was found to be publicly accessible.** This confirms the CDC's own framing (section 6/35) that production must run in a clearly labelled DEMO/PROXY mode until STEG/ANME can supply real observations.

## E — Real production data from a compatible research/reference dataset (NOT measured, but real physical/satellite data)

These are legitimate, citable, real-data-driven **estimates**, not measurements — CDC section 35 explicitly forbids presenting them as "measured". They matter because they are a much stronger reference than an ad hoc synthetic generator: they're grounded in real satellite-derived irradiance climatology (SARAH2/ERA5), independently validated, and used across the energy-research industry.

| Source | Basis | Reachability from this environment | Status |
|---|---|---|---|
| **PVGIS** (`re.jrc.ec.europa.eu`, JRC European Commission) — `seriescalc`/PVcalc | Real satellite irradiance (SARAH2 for Africa/Europe) + physical PV model, per-coordinate hourly time series | Tested directly: request reached the server (HTTP 200) but was **rejected by PVGIS's own WAF/bot-protection** ("The requested URL was rejected") when called through this environment's tooling. Not reachable here; likely reachable from a normal browser/server IP. | **Reachable in principle, blocked here — recommended primary reference source once run outside this sandbox** |
| **Renewables.ninja** (renewables.ninja) | MERRA-2 reanalysis-driven PV/wind simulation, widely cited in academic energy research | Requires account/API token; not tested here (no token available) | **Reachable in principle with a (free) API token — not tested** |
| **NASA POWER** | Satellite-derived solar irradiance, used e.g. by profilesolar.com's Tunisia PV-potential pages (46 locations) | Not tested directly; third-party site (profilesolar.com) confirms it is usable for Tunisia | **Likely reachable — not tested directly, good candidate** |

**Recommendation:** replace this repo's `SyntheticClearSkyProvider`-driven proxy target with a **PVGIS-driven** reference simulation as soon as this pipeline runs outside the current network-restricted build sandbox — same "reference, not measured" caveat still applies, but it removes the "invented from scratch" character of the current fallback and is directly aligned with CDC section 3's request to evaluate PVGIS.

## F — Fallback position adopted in this repo (per CDC section 5.F)

No real production source exists → the repository keeps a physics-based proxy target, but:
- it is generated **only** in `DATA_MODE=demo` (see `configs/config.yaml` and `src/config_mode.py`),
- every row is tagged `pv_production_source=DEMO_PROXY_PHYSICS_SIMULATION`,
- `DATA_MODE=real` **refuses to run** the training/forecast pipeline and raises `RealDataUnavailableError` naming exactly which source is missing (see `src/ingestion/data_mode.py`),
- no forecasting/uncertainty/backtesting metric produced against this proxy is ever described as validated real-world performance anywhere in this repo's docs.

## What would unlock real district-level supervised training

1. STEG/ANME providing a Prosol self-consumption/injection production export (even monthly aggregated per governorate would materially help — see §5.C/D in the CDC).
2. Re-running `src/ingestion/pvgis_provider.py` (to be added) from an environment where `re.jrc.ec.europa.eu` is reachable, to get a real-satellite-irradiance-driven reference for all 50 districts.
3. A Renewables.ninja API token for a second, independently-computed cross-check.
