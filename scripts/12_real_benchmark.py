"""
Script 12 - REAL-DATA benchmark on measured ENSTAB production (MEASURED_REAL).

Fills Tables 1 & 2 of the technical report with *measured* numbers:
  baselines   : persistence, smart persistence (clear-sky-index), climatology
  physics     : clear-sky, national config (uncalibrated), calibrated (measured GHI)
  ML          : LightGBM, XGBoost   (direct, per horizon)
  uncertainty : quantile LightGBM P10/P90 + split-conformal (CQR), coverage per horizon
  extras      : feature-group ablation, SHAP, figures, saved models, LaTeX tables

Run (about 2-4 min on a laptop):
    python scripts/12_real_benchmark.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecasting import real_pv as R  # noqa: E402

OUT = Path("data/validation")
FIG = Path("reports/figures")
MODELS = Path("models/registry/real_enstab")
for d in (OUT, FIG, MODELS, Path("reports/explainability"), Path("configs")):
    d.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("Loading ENSTAB (measured) ...")
df = R.load_enstab()
tr_mask = (df.index >= R.SPLITS["train"][0]) & (df.index <= pd.Timestamp(R.SPLITS["train"][1]) + pd.Timedelta(days=1))
c_pcs = R.fit_clearsky_power_constant(df[tr_mask])
df = R.add_targets(df, c_pcs)
feats = R.base_features(df)
cal = R.fit_calibrated_physics(df[tr_mask])
print(f"  rows={len(df):,}  c_pcs={c_pcs:.3f} W per W/m2   calibrated physics: {cal}")

# ------------------------------------------------------------ physics diagnostics
te = df[(df.index >= R.SPLITS["test"][0]) & (df.index <= pd.Timestamp(R.SPLITS["test"][1]) + pd.Timedelta(days=1))]
frac_te = float(te["day"].mean())
dd = te[te["day"]]
phys_rows = []
for name, ph in [
    ("Clear-sky physics (no cloud info)", dd["pcs"].values),
    ("National config, UNcalibrated (PR=0.80) + measured GHI", R.national_config_physics(dd["ghi"].values, dd["t_amb"].values)),
    ("Calibrated physics (ENSTAB-fit) + measured GHI & Tcell", R.physics_power(dd["ghi"].values, dd["t_cell"].values, cal)),
]:
    phys_rows.append(dict(model=name, **R.metrics(dd["p"].values, ph, frac_te)))
phys = pd.DataFrame(phys_rows)
phys.to_csv(OUT / "real_physics_transfer.csv", index=False)
print(phys[["model", "MAE_W", "nMAE_pct", "bias_W", "R2"]].round(2).to_string(index=False))

# ------------------------------------------------------------ per-horizon benchmark
rows, cov_rows, abl_rows, shap_done = [], [], [], set()
PRESETS = {}
fc_store = []
clim_cache = {}

for hz in R.HORIZONS:
    X = R.build_dataset(df, feats, hz)
    trn = X[R.split_mask(X["t"], "train") & X["day"] & X["y"].notna()]
    val = X[R.split_mask(X["t"], "val") & X["day"] & X["y"].notna()]
    tst_all = X[R.split_mask(X["t"], "test")]
    tst = tst_all[tst_all["day"]].copy()
    frac = float(tst_all["day"].mean())
    w_tr = (trn["pcs"] / R.PNOM_W).values
    F = R.FEATURE_SETS["full"]

    # ---- baselines (all forced physical: night = 0, 0..Pnom)
    table = R.climatology_table(trn)
    preds = {
        "Persistence": R.apply_constraints(tst["p_persist"].values, tst["pcs"].values),
        "Smart persistence (clear-sky index)": R.apply_constraints(
            tst["pcs"].values * np.nan_to_num(tst["kt_issue"].values, nan=0.5), tst["pcs"].values),
        "Climatology (month x hour)": R.apply_constraints(R.climatology_pred(tst, table), tst["pcs"].values),
    }
    # ---- ML
    w_va = (val["pcs"] / R.PNOM_W).values
    preset = R.select_preset(trn[F], trn["y"], w_tr, val[F], val["y"].values, w_va)
    ex = R.LGB_PRESETS[preset]
    m_l = R.fit_lgbm(trn[F], trn["y"], w_tr, extra=ex)
    m_x = R.fit_xgb(trn[F], trn["y"], w_tr)
    preds["LightGBM"] = R.apply_constraints(m_l.predict(tst[F]) * tst["pcs"].values, tst["pcs"].values)
    preds["XGBoost"] = R.apply_constraints(m_x.predict(tst[F]) * tst["pcs"].values, tst["pcs"].values)
    # ---- quantiles + conformal (calibrated on VALIDATION only)
    q10 = R.fit_lgbm(trn[F], trn["y"], w_tr, "quantile", 0.10, extra=ex)
    q90 = R.fit_lgbm(trn[F], trn["y"], w_tr, "quantile", 0.90, extra=ex)
    lo_v, hi_v = np.sort(np.c_[q10.predict(val[F]), q90.predict(val[F])], axis=1).T
    qhat = R.conformal_qhat(lo_v, hi_v, val["y"].values, alpha=0.2)
    lo_t, hi_t = np.sort(np.c_[q10.predict(tst[F]), q90.predict(tst[F])], axis=1).T
    pcs = tst["pcs"].values
    ints = {"raw": (lo_t, hi_t), "conformal": (lo_t - qhat, hi_t + qhat)}
    for kind, (lo, hi) in ints.items():
        lo_w = R.apply_constraints(np.maximum(lo, 0) * pcs, pcs)
        hi_w = R.apply_constraints(hi * pcs, pcs)
        inside = (tst["p"].values >= lo_w - 1e-6) & (tst["p"].values <= hi_w + 1e-6)
        cov_rows.append(dict(horizon=hz, interval=kind, nominal_pct=80.0,
                             empirical_pct=100 * inside.mean(),
                             mean_width_pct_of_pnom=100 * np.mean(hi_w - lo_w) / R.PNOM_W))
        if kind == "conformal":
            conf_lo, conf_hi = lo_w, hi_w

    base_mae = {k: np.mean(np.abs(v - tst["p"].values)) for k, v in preds.items()}
    for name, ph in preds.items():
        m = R.metrics(tst["p"].values, ph, frac)
        m.update(horizon=hz, model=name,
                 skill_vs_persistence_pct=100 * (1 - m["MAE_W"] / base_mae["Persistence"]),
                 skill_vs_smart_pct=100 * (1 - m["MAE_W"] / base_mae["Smart persistence (clear-sky index)"]))
        rows.append(m)

    # ---- ablation (LightGBM only, 4 horizons)
    if hz in ("15min", "3h", "J+1", "J+3"):
        for fs in ("full", "no_met", "no_history"):
            cols = R.FEATURE_SETS[fs]
            mm = m_l if fs == "full" else R.fit_lgbm(trn[cols], trn["y"], w_tr, extra=ex)
            ph = R.apply_constraints(mm.predict(tst[cols]) * pcs, pcs)
            abl_rows.append(dict(horizon=hz, feature_set=fs, **R.metrics(tst["p"].values, ph, frac)))

    # ---- SHAP (2 horizons)
    if hz in ("15min", "J+1"):
        import shap
        samp = tst.sample(min(1500, len(tst)), random_state=0)
        sv = shap.TreeExplainer(m_l).shap_values(samp[F])
        s = pd.Series(np.abs(sv).mean(0), index=F).sort_values(ascending=False)
        s.rename("mean_abs_shap").to_csv(f"reports/explainability/real_shap_{hz}.csv")
        shap_done.add(hz)

    # ---- store test forecasts (all test target rows incl. night) + models
    allp = tst_all.copy()
    allp["pcs"] = allp["pcs"].values
    out = pd.DataFrame({"horizon": hz, "timestamp": allp["t"].values, "issue_time": allp["issue"].values,
                        "actual_w": allp["p"].values})
    idx_day = allp["day"].values
    for key, name in [("persistence_w", "Persistence"), ("smart_w", "Smart persistence (clear-sky index)"),
                      ("climatology_w", "Climatology (month x hour)"), ("lgbm_w", "LightGBM"), ("xgb_w", "XGBoost")]:
        out[key] = 0.0
        out.loc[idx_day, key] = preds[name]
    out["p10_w"], out["p90_w"] = 0.0, 0.0
    out.loc[idx_day, "p10_w"], out.loc[idx_day, "p90_w"] = conf_lo, conf_hi
    out["clearsky_w"] = allp["pcs"].values
    num = out.select_dtypes('number').columns
    out[num] = out[num].round(1)
    fc_store.append(out)
    joblib.dump({"mean": m_l, "q10": q10, "q90": q90, "qhat": qhat, "features": F}, MODELS / f"{hz}.joblib", compress=3)
    cur = {r["model"]: r["nMAE_pct"] for r in rows if r["horizon"] == hz}
    print(f"  {hz:>5}: preset={preset:<14} LGBM={cur['LightGBM']:.2f}%  smart={cur['Smart persistence (clear-sky index)']:.2f}%  "
          f"clim={cur['Climatology (month x hour)']:.2f}%  conf cov={cov_rows[-1]['empirical_pct']:.1f}%  ({time.time()-t0:.0f}s)")
    PRESETS[hz] = preset

M = pd.DataFrame(rows)
C = pd.DataFrame(cov_rows)
A = pd.DataFrame(abl_rows)
M.to_csv(OUT / "real_benchmark_metrics.csv", index=False)
C.to_csv(OUT / "real_benchmark_coverage.csv", index=False)
A.to_csv(OUT / "real_benchmark_ablation.csv", index=False)
pd.concat(fc_store, ignore_index=True).to_csv(OUT / "real_test_forecasts.csv.gz", index=False, compression="gzip")
R.save_json({"c_pcs": c_pcs, "physics_calibration": cal, "pnom_w": R.PNOM_W, "site": R.SITE,
             "lgbm_presets": PRESETS, "splits": R.SPLITS, "issue_hour_local": R.ISSUE_HOUR,
             "data_class": "MEASURED_REAL_ENSTAB"}, "configs/calibration_enstab.json")
R.save_json({"horizons": list(R.HORIZONS), "features": R.FEATURE_SETS["full"], "c_pcs": c_pcs},
            MODELS / "meta.json")

# ------------------------------------------------------------ figures
order = list(R.HORIZONS)
colors = {"Persistence": "#999", "Smart persistence (clear-sky index)": "#e69f00",
          "Climatology (month x hour)": "#56b4e9", "LightGBM": "#d55e00", "XGBoost": "#009e73"}
fig, ax = plt.subplots(figsize=(8, 4.2))
for name, col in colors.items():
    s = M[M.model == name].set_index("horizon").loc[order]
    ax.plot(order, s["nMAE_pct"], marker="o", label=name, color=col)
ax.set(ylabel="nMAE, daytime (% of Pnom)", title="Measured ENSTAB test set (Dec 2023 - May 2024)")
ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "fig1_nmae_by_horizon.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
for kind, ls in [("raw", "--"), ("conformal", "-")]:
    s = C[C.interval == kind].set_index("horizon").loc[order]
    ax.plot(order, s["empirical_pct"], ls, marker="o", label=f"P10-P90 {kind}")
ax.axhline(80, color="k", lw=1, ls=":", label="nominal 80%")
ax.set(ylabel="empirical coverage (%)", ylim=(40, 100), title="Interval reliability by horizon"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "fig2_coverage.png", dpi=160); plt.close(fig)

F_ = pd.concat(fc_store, ignore_index=True)
F_["timestamp"] = pd.to_datetime(F_["timestamp"])
day_tot = F_[F_.horizon == "J+1"].groupby(F_["timestamp"].dt.date)["actual_w"].sum()
clear_day, cloudy_day = day_tot.idxmax(), day_tot[day_tot > day_tot.quantile(.1)].idxmin()
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
for r, hz in enumerate(["15min", "J+1"]):
    for c, (lab, day) in enumerate([("clear day", clear_day), ("cloudy day", cloudy_day)]):
        g = F_[(F_.horizon == hz) & (F_.timestamp.dt.date == day)]
        a = axes[r, c]
        a.fill_between(g.timestamp, g.p10_w, g.p90_w, alpha=.25, color="#d55e00", label="P10-P90 (conformal)")
        a.plot(g.timestamp, g.actual_w, "k", lw=1.5, label="measured")
        a.plot(g.timestamp, g.lgbm_w, color="#d55e00", lw=1.2, label="LightGBM")
        a.plot(g.timestamp, g.persistence_w, color="#999", lw=1, ls="--", label="persistence")
        a.set_title(f"{hz} - {lab} ({day})", fontsize=9); a.tick_params(axis="x", labelrotation=30, labelsize=7)
axes[0, 0].legend(fontsize=7); axes[0, 0].set_ylabel("W"); axes[1, 0].set_ylabel("W")
fig.tight_layout(); fig.savefig(FIG / "fig3_examples.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for a, hz in zip(axes, ["15min", "J+1"]):
    s = pd.read_csv(f"reports/explainability/real_shap_{hz}.csv", index_col=0).iloc[:10, 0][::-1]
    a.barh(s.index, s.values, color="#0072b2"); a.set_title(f"SHAP mean |value| - {hz}", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig4_shap.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
w = 0.26
for i, fs in enumerate(["full", "no_met", "no_history"]):
    s = A[A.feature_set == fs].set_index("horizon").loc[["15min", "3h", "J+1", "J+3"]]
    ax.bar(np.arange(4) + (i - 1) * w, s["nMAE_pct"], w, label=fs)
ax.set_xticks(range(4)); ax.set_xticklabels(["15min", "3h", "J+1", "J+3"]); ax.set_ylabel("nMAE daytime (%)")
ax.set_title("Ablation: what is the model really using?"); ax.legend(); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(FIG / "fig5_ablation.png", dpi=160); plt.close(fig)

# ------------------------------------------------------------ report + LaTeX tables
j1 = M[M.horizon == "J+1"].set_index("model")
t1 = j1.loc[["Persistence", "Smart persistence (clear-sky index)", "LightGBM", "XGBoost"],
            ["MAE_W", "RMSE_W", "nMAE_pct", "bias_W", "skill_vs_persistence_pct"]].round(2)
_p = phys.set_index("model")[["MAE_W", "RMSE_W", "nMAE_pct", "bias_W"]].round(2)
_p.index = [i + " [same-time diagnostic, not a forecast]" for i in _p.index]
t1 = pd.concat([t1, _p.iloc[[0, 2]]])
t2 = M[M.model == "LightGBM"].set_index("horizon").loc[order, ["MAE_W", "RMSE_W", "nMAE_pct", "nMAE24_pct"]].round(2)
t2["P10-P90 coverage (%)"] = C[C.interval == "conformal"].set_index("horizon").loc[order, "empirical_pct"].round(1)
t2["raw coverage (%)"] = C[C.interval == "raw"].set_index("horizon").loc[order, "empirical_pct"].round(1)

md = f"""# REAL-DATA BENCHMARK (MEASURED_REAL - ENSTAB Borj Cedria)

*Auto-generated by `scripts/12_real_benchmark.py`. Every number below is computed against **measured** PV power.*

- Data: 15-min means of 5-min ENSTAB measurements, 2022-02-24 -> 2024-05-31 (~3 kWc rooftop system, capacity *estimated*)
- Chronological split by target time: train {R.SPLITS['train']} | val {R.SPLITS['val']} | **test {R.SPLITS['test']}**
- Metrics are on **daytime** timesteps (clear-sky power > {R.DAY_PCS_MIN_W:.0f} W), nMAE normalised by Pnom = {R.PNOM_W:.0f} W.
  `nMAE24` re-includes night (zeros) for comparison with 24 h-averaged numbers.
- Day-ahead models use **no NWP forecast** (none exists for this site): measured history + astronomy only. They are a *floor*.

## Table 1 - external validation on real PV, horizon J+1 (test set)
{t1.to_markdown()}

## Table 2 - LightGBM performance and interval coverage per horizon (test set)
{t2.to_markdown()}

## All models, all horizons (nMAE daytime, % of Pnom)
{M.pivot(index='model', columns='horizon', values='nMAE_pct')[order].round(2).to_markdown()}

## Skill vs smart persistence (% MAE reduction; >0 = better)
{M.pivot(index='model', columns='horizon', values='skill_vs_smart_pct')[order].round(1).to_markdown()}

## Interval reliability
{C.round(2).to_markdown(index=False)}

## Physics-model transfer test (same-time diagnostic with *measured* GHI - NOT a forecast)
{phys.round(2)[['model','MAE_W','nMAE_pct','bias_W','R2']].to_markdown(index=False)}

## Ablation (LightGBM nMAE daytime %)
{A.pivot(index='feature_set', columns='horizon', values='nMAE_pct').round(2).to_markdown()}

Figures: `reports/figures/fig1..fig5`. SHAP tables: `reports/explainability/real_shap_*.csv`.
Calibrated physics: A={cal['A_w_per_wm2']:.3f} W per W/m2, gamma={cal['gamma_per_c']*100:.3f} %/C (`configs/calibration_enstab.json`).
"""
Path("reports/REAL_FORECAST_BENCHMARK.md").write_text(md)
tex = "% Auto-generated. Paste into the LaTeX report.\n" + t1.to_latex(float_format="%.2f") + "\n" + t2.to_latex(float_format="%.2f")
Path("reports/report_tables.tex").write_text(tex)
print(f"Done in {time.time()-t0:.0f}s -> reports/REAL_FORECAST_BENCHMARK.md")
