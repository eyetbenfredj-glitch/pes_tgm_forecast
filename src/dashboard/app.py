"""
Streamlit dashboard (CDC continuation Phase 16/17). 10 pages + interactive
Tunisia map. Every page shows a REAL DATA / DEMO-SYNTHETIC DATA banner driven
by DATA_MODE and the dataset's own provenance columns - never silently
presented as validated real-world output.

Run:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.aggregation.hierarchical import reconcile_all_levels, check_consistency
from src.monitoring.drift import weather_drift_report, production_drift_report, anomaly_flags
from src.forecasting.backtesting import evaluate
from src.ingestion.data_mode import get_data_mode, DataMode
from src.forecasting import national_hybrid as NH

st.set_page_config(page_title="PESTGM Track 1 - PV Forecast (Tunisia)", layout="wide")

DATA_PATH = Path("data/processed/training_dataset_15min_real_reference_sample.csv")
TARGET_COL = "pv_production_mw_reference"

if not DATA_PATH.exists():
    DATA_PATH = Path("data/processed/training_dataset_15min_sample.csv")
    TARGET_COL = "pv_production_mw_proxy"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df[NH.FCOL] = NH.hybrid_forecast_mw(df)
    return df


VAL = Path("data/validation")


@st.cache_data
def load_real():
    out = {}
    for k, f in [("m", "real_benchmark_metrics.csv"), ("c", "real_benchmark_coverage.csv"),
                 ("a", "real_benchmark_ablation.csv"), ("p", "real_physics_transfer.csv")]:
        out[k] = pd.read_csv(VAL / f) if (VAL / f).exists() else None
    fp = VAL / "real_test_forecasts.csv.gz"
    out["f"] = pd.read_csv(fp, parse_dates=["timestamp"]) if fp.exists() else None
    return out


@st.cache_resource
def _real_predict():
    from src.forecasting import real_service
    return real_service


def provenance_banner(df: pd.DataFrame):
    mode = get_data_mode()
    weather_src = df["weather_source"].iloc[0] if "weather_source" in df.columns else "UNKNOWN"
    src_col = "production_source" if TARGET_COL == "pv_production_mw_reference" else "pv_production_source"
    prod_src = df[src_col].iloc[0] if src_col in df.columns else "UNKNOWN"
    is_real = (mode == DataMode.REAL)
    if is_real:
        st.success("REAL DATA — DATA_MODE=real")
    else:
        st.warning(
            f"DEMO / SYNTHETIC DATA — DATA_MODE=demo | weather_source={weather_src} | "
            f"production_source={prod_src}. Not validated against real STEG production. "
            f"See docs/REAL_PRODUCTION_DATA_SOURCES.md."
        )


df = load_data()

st.sidebar.title("PESTGM 7.0 — Track 1")
page = st.sidebar.radio("Page", [
    "1. National overview", "2. Governorate overview", "3. District forecast",
    "4. Uncertainty", "5. Forecast vs observed", "6. Forecast errors",
    "7. Weather", "8. Monitoring / anomalies", "9. Model / explainability",
    "10. Data provenance / status", "Tunisia map",
    "11. REAL validation (ENSTAB measured)", "12. Live forecast (real model replay)",
])

provenance_banner(df)

# ---------------------------------------------------------------- Page 1 --
if page == "1. National overview":
    st.header("National overview")
    levels = reconcile_all_levels(df, TARGET_COL)
    national = levels["national"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total installed capacity (MW)", f"{df.drop_duplicates('district_id')['capacity_mw'].sum():.1f}")
    col2.metric("Peak national forecast (MW, DEMO)", f"{national[TARGET_COL].max():.1f}")
    col3.metric("Districts", df["district_id"].nunique())
    national = national.merge(reconcile_all_levels(df, NH.FCOL)["national"], on="timestamp")
    fig = px.line(national, x="timestamp", y=[NH.FCOL, TARGET_COL],
                  title="National PV: ENSTAB-calibrated physics forecast vs simulated reference (DEMO)")
    st.plotly_chart(fig, width="stretch")
    consistency = check_consistency(levels, TARGET_COL)
    st.caption(f"Hierarchical consistency: {consistency}")

# ---------------------------------------------------------------- Page 2 --
elif page == "2. Governorate overview":
    st.header("Governorate overview")
    levels = reconcile_all_levels(df, TARGET_COL)
    gov = st.selectbox("Governorate", sorted(df["governorate"].unique()))
    g = levels["governorate"][levels["governorate"]["governorate"] == gov]
    g = g.merge(reconcile_all_levels(df[df["governorate"] == gov], NH.FCOL)["governorate"], on=["timestamp", "governorate"])
    fig = px.line(g, x="timestamp", y=[NH.FCOL, TARGET_COL], title=f"{gov} — forecast vs simulated reference")
    st.plotly_chart(fig, width="stretch")
    districts_here = df[df["governorate"] == gov].drop_duplicates("district_id")
    st.dataframe(districts_here[["district_id", "district", "capacity_mw"]].sort_values("capacity_mw", ascending=False))

# ---------------------------------------------------------------- Page 3 --
elif page == "3. District forecast":
    st.header("District forecast")
    district = st.selectbox("District", sorted(df["district"].unique()))
    g = df[df["district"] == district].sort_values("timestamp")
    fig = px.line(g, x="timestamp", y=[NH.FCOL, TARGET_COL], title=f"{district} — forecast vs simulated reference")
    fig.add_hline(y=g["capacity_mw"].iloc[-1], line_dash="dot", annotation_text="installed capacity")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(g[["timestamp", TARGET_COL, "capacity_mw", "shortwave_radiation", "temperature_2m"]].tail(20))

# ---------------------------------------------------------------- Page 4 --
elif page == "4. Uncertainty":
    st.header("Uncertainty (P10-P50-P90)")
    st.caption("P50 = ENSTAB-calibrated physics forecast. Band = interval TRANSFERRED from the ENSTAB "
               "conformal-calibrated J+1 ratios (single site, conservative for aggregates). "
               "True per-horizon coverage on measured data: page 11.")
    district = st.selectbox("District", sorted(df["district"].unique()), key="unc_district")
    horizon = st.selectbox("Horizon whose error profile to transfer", ["15min", "1h", "3h", "6h", "J+1", "J+2", "J+3"], index=4)
    g = df[df["district"] == district].sort_values("timestamp").tail(200).copy()
    iv = NH.add_interval(g[NH.FCOL], g["capacity_mw"], horizon)
    g = pd.concat([g, iv[["p10_mw", "p90_mw"]]], axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["timestamp"], y=g["p90_mw"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=g["timestamp"], y=g["p10_mw"], fill="tonexty", line=dict(width=0), name="P10-P90 band"))
    fig.add_trace(go.Scatter(x=g["timestamp"], y=g[NH.FCOL], name="P50 forecast", line=dict(color="black")))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Page 5 --
elif page == "5. Forecast vs observed":
    st.header("Forecast vs simulated reference (demo) - real measured comparison: page 11")
    st.caption("No real observed production exists yet (see docs/REAL_PRODUCTION_DATA_SOURCES.md). "
               "This page compares the persistence baseline forecast against the DEMO_PROXY target "
               "to demonstrate the intended real-vs-forecast comparison view.")
    district = st.selectbox("District", sorted(df["district"].unique()), key="fvo_district")
    g = df[df["district"] == district].sort_values("timestamp").copy()
    g["persistence_forecast"] = g[TARGET_COL].shift(1)
    g = g.tail(200)
    fig = px.line(g, x="timestamp", y=[TARGET_COL, NH.FCOL, "persistence_forecast"],
                  labels={"value": "MW"}, title=f"{district}: simulated reference vs hybrid forecast vs persistence")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Page 6 --
elif page == "6. Forecast errors":
    st.header("Forecast errors (persistence baseline, DEMO_ONLY)")
    district = st.selectbox("District", sorted(df["district"].unique()), key="err_district")
    g = df[df["district"] == district].sort_values("timestamp").copy()
    g["persistence_forecast"] = g[TARGET_COL].shift(1)
    g = g.dropna(subset=["persistence_forecast"])
    m = evaluate(g[TARGET_COL], g["persistence_forecast"], g["capacity_mw"])
    st.json(m)
    g["error"] = g["persistence_forecast"] - g[TARGET_COL]
    fig = px.histogram(g, x="error", title="Error distribution (MW)")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Page 7 --
elif page == "7. Weather":
    st.header("Weather")
    district = st.selectbox("District", sorted(df["district"].unique()), key="wx_district")
    g = df[df["district"] == district].sort_values("timestamp").tail(200)
    fig = px.line(g, x="timestamp", y=["shortwave_radiation", "temperature_2m", "cloud_cover"],
                  title=f"{district} — weather variables ({g['weather_source'].iloc[0]})")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Page 8 --
elif page == "8. Monitoring / anomalies":
    st.header("Monitoring / anomalies")
    ref_end = df["timestamp"].quantile(0.5)
    st.subheader("Weather drift (PSI)")
    st.dataframe(weather_drift_report(df, ref_end))
    st.subheader("Production drift (PSI)")
    st.json(production_drift_report(df, TARGET_COL, ref_end))
    st.subheader("Physical anomaly flags")
    flags = anomaly_flags(df, production_col=TARGET_COL)
    pct = 100 * flags["physical_anomaly_flag"].mean()
    st.metric("% rows flagged anomalous", f"{pct:.2f}%")
    st.dataframe(flags[flags["physical_anomaly_flag"]].head(50))

# ---------------------------------------------------------------- Page 9 --
elif page == "9. Model / explainability":
    st.header("Model / explainability")
    exp_path = Path("reports/explainability/global_feature_importance.csv")
    if exp_path.exists():
        gfi = pd.read_csv(exp_path)
        fig = px.bar(gfi.head(10), x="mean_abs_shap", y="feature", orientation="h",
                     title="Global feature importance (SHAP, LightGBM)")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Run `python -c \"...\"` (see README) or the explainability module to generate "
                "reports/explainability/global_feature_importance.csv first.")
    st.caption("Model trained on DEMO_PROXY_PHYSICS_SIMULATION - see reports/MODEL_VALIDATION_REPORT.md")

# --------------------------------------------------------------- Page 10 --
elif page == "10. Data provenance / status":
    st.header("Data provenance / status")
    st.write(f"DATA_MODE: **{get_data_mode().value}**")
    prov = df[["weather_source", "pv_production_source", "capacity_source"]].drop_duplicates()
    st.dataframe(prov)
    st.markdown("See `docs/DATA_READINESS_REPORT.md` and `docs/REAL_PRODUCTION_DATA_SOURCES.md` "
                "for the full real/derived/model-output/demo-only classification.")

# ------------------------------------------------------------------ Map --
elif page == "Tunisia map":
    st.header("Tunisia — district map (representative points, not official polygons)")
    metric = st.selectbox("Show", ["capacity_mw", TARGET_COL])
    latest = df.sort_values("timestamp").groupby("district_id").tail(1)
    fig = px.scatter_map(
        latest, lat="latitude", lon="longitude", size=metric, color=metric,
        hover_name="district", hover_data=["governorate", "capacity_mw", TARGET_COL],
        zoom=5.3, height=650, map_style="open-street-map",
        title="Representative district locations (not official administrative boundaries)",
    )
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------- Page 11 --
elif page == "11. REAL validation (ENSTAB measured)":
    st.header("Real-world validation on MEASURED production (ENSTAB Borj Cedria)")
    st.success("MEASURED_REAL_ENSTAB - every number on this page is computed against measured PV power, "
               "on a chronological hold-out (Dec 2023 - May 2024) never seen in training.")
    rd = load_real()
    if rd["m"] is None:
        st.error("Run `python scripts/12_real_benchmark.py` first."); st.stop()
    M, C = rd["m"], rd["c"]
    order = ["15min", "1h", "3h", "6h", "J+1", "J+2", "J+3"]
    j1 = M[(M.horizon == "J+1")].set_index("model")
    cj = C[(C.horizon == "J+1") & (C.interval == "conformal")].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("J+1 nMAE (LightGBM)", f"{j1.loc['LightGBM','nMAE_pct']:.1f}%", help="daytime, % of Pnom")
    c2.metric("15-min nMAE", f"{M[(M.horizon=='15min')&(M.model=='LightGBM')].nMAE_pct.iloc[0]:.1f}%")
    c3.metric("Skill vs persistence (J+1)", f"{j1.loc['LightGBM','skill_vs_persistence_pct']:.0f}%")
    c4.metric("P10-P90 coverage (J+1)", f"{cj.empirical_pct:.0f}%", help="nominal 80%")

    st.subheader("Error by horizon and model")
    piv = M.pivot(index="horizon", columns="model", values="nMAE_pct").loc[order]
    st.plotly_chart(px.line(piv, markers=True, labels={"value": "nMAE daytime (% of Pnom)", "horizon": "horizon"}),
                    width="stretch")
    st.dataframe(piv.round(2))
    st.caption("Day-ahead models use NO weather forecast (none exists for this site) - they are a floor. "
               "Beyond J+1 they only match climatology: this is the quantified case for adding NWP irradiance.")

    st.subheader("Forecast vs measured")
    F = rd["f"]
    hz = st.selectbox("Horizon", order, index=4)
    days = sorted(F[F.horizon == hz].timestamp.dt.date.unique())
    day = st.select_slider("Day (test period)", options=days, value=days[len(days) // 2])
    g = F[(F.horizon == hz) & (F.timestamp.dt.date == day)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.timestamp, y=g.p90_w, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=g.timestamp, y=g.p10_w, fill="tonexty", line=dict(width=0), name="P10-P90 (conformal)"))
    fig.add_trace(go.Scatter(x=g.timestamp, y=g.actual_w, name="measured", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=g.timestamp, y=g.lgbm_w, name="LightGBM", line=dict(color="#d55e00")))
    fig.add_trace(go.Scatter(x=g.timestamp, y=g.persistence_w, name="persistence", line=dict(color="#999", dash="dash")))
    fig.update_layout(yaxis_title="W", height=420)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Uncertainty reliability (nominal 80%)")
    st.plotly_chart(px.line(C, x="horizon", y="empirical_pct", color="interval", markers=True,
                            category_orders={"horizon": order}), width="stretch")
    if rd["a"] is not None:
        st.subheader("Ablation - what the model really uses")
        st.dataframe(rd["a"].pivot(index="feature_set", columns="horizon", values="nMAE_pct").round(2))
    if rd["p"] is not None:
        st.subheader("Physics transfer test (same-time, measured GHI - diagnostic, not a forecast)")
        st.dataframe(rd["p"][["model", "MAE_W", "nMAE_pct", "bias_W", "R2"]].round(2))
        st.caption("The uncalibrated national config (PR=0.80) under-predicts the real system; calibrating on ENSTAB removes the bias. "
                   "Caveat: implied effective PR is ~0.93 at the estimated 3 kWc - nameplate and irradiance-sensor orientation must be confirmed.")

# --------------------------------------------------------------- Page 12 --
elif page == "12. Live forecast (real model replay)":
    st.header("Live forecast - trained model, run on demand")
    st.success("MEASURED_REAL_ENSTAB replay: features are rebuilt from measurements up to the issue time ONLY; "
               "the trained LightGBM + conformal quantile models then produce the forecast. Nothing after the issue time is used.")
    svc = _real_predict()
    lo, hi = svc.available_issue_range()
    c1, c2 = st.columns(2)
    d = c1.date_input("Issue day", value=pd.Timestamp("2024-04-10"), min_value=pd.Timestamp("2022-03-10"),
                      max_value=pd.Timestamp("2024-05-27"))
    hz = c2.selectbox("Horizon", ["J+1", "J+2", "J+3", "15min", "1h", "3h", "6h"])
    issue = pd.Timestamp(d) + pd.Timedelta(hours=10)
    if hz in ("15min", "1h", "3h", "6h"):
        hh = st.slider("Issue hour (local)", 0.0, 23.75, 10.0, 0.25)
        issue = pd.Timestamp(d) + pd.Timedelta(hours=hh)
    try:
        rows = pd.DataFrame(svc.predict(str(issue), hz))
    except Exception as e:
        st.error(str(e)); st.stop()
    rows["target_time"] = pd.to_datetime(rows["target_time"])
    st.caption(f"Forecast issued {rows.issue_time.iloc[0]} for {rows.target_time.min()} -> {rows.target_time.max()}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rows.target_time, y=rows.p90_w, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=rows.target_time, y=rows.p10_w, fill="tonexty", line=dict(width=0), name="P10-P90"))
    fig.add_trace(go.Scatter(x=rows.target_time, y=rows.clearsky_w, name="clear-sky", line=dict(color="#bbb", dash="dot")))
    fig.add_trace(go.Scatter(x=rows.target_time, y=rows.forecast_w, name="forecast", line=dict(color="#d55e00")))
    fig.add_trace(go.Scatter(x=rows.target_time, y=rows.measured_w, name="measured (revealed afterwards)", line=dict(color="black")))
    fig.update_layout(yaxis_title="W", height=430)
    st.plotly_chart(fig, width="stretch")
    dd = rows[rows.clearsky_w > 20]
    if len(dd) > 0:
        mae = (dd.forecast_w - dd.measured_w).abs().mean()
        cov = ((dd.measured_w >= dd.p10_w) & (dd.measured_w <= dd.p90_w)).mean() * 100
        a, b = st.columns(2)
        a.metric("MAE of this forecast (daytime)", f"{mae:.0f} W")
        b.metric("Measured inside P10-P90", f"{cov:.0f}%")
