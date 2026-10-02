"""
Continuous learning / model update workflow (CDC continuation Phase 11).

Explicit steps, each one actually executed (not just described):
  1. ingest new observations (a batch treated as "newly arrived")
  2. validate them (physical checks)
  3. compute forecast error vs the current raw model
  4. update the residual-correction model on that error
  5. recalibrate the uncertainty (conformal offsets) on the same batch
  6. decide whether to retrain the base model (simple, documented rule:
     retrain if rolling MAE degraded > model_degradation_threshold vs the
     model's own training-time validation MAE)
  7. save a new versioned model + calibration artifact under models/registry/

DEMO_ONLY: the "new observations" batch here is the existing test split of
the labelled DEMO/PROXY dataset, used to demonstrate the mechanism actually
running end-to-end. In production this would be triggered by a real new data
arrival (CLI/cron/API call - see --trigger flag below for the three
supported invocation modes).

Usage:
    python scripts/07_update_forecast_model.py --trigger cli
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.data_mode import get_data_mode, DataMode
from src.forecasting.backtesting import make_chrono_split, evaluate
from src.models.baselines import LightGBMBaseline
from src.calibration.residual_correction import RollingBiasCorrector
from src.uncertainty.quantile_forecast import QuantileForecastModel, evaluate_coverage
from src.validation.physical_checks import run_all_checks

DATA_PATH = Path("data/processed/training_dataset_15min_real_reference.csv")
TARGET_COL = "pv_production_mw_reference"
if not DATA_PATH.exists():
    DATA_PATH = Path("data/processed/training_dataset_15min.csv")
    TARGET_COL = "pv_production_mw_proxy"

REGISTRY_DIR = Path("models/registry")
DEGRADATION_THRESHOLD = 1.3  # retrain if new-batch MAE > 1.3x training-time validation MAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", choices=["cli", "api", "scheduled"], default="cli")
    args = ap.parse_args()

    mode = get_data_mode()
    if mode == DataMode.REAL:
        print("DATA_MODE=real: refusing - no real production stream configured yet.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    split = make_chrono_split(df, train_frac=0.6, val_frac=0.2)
    train, val, test = split.split(df)

    # ---- step 0: base model "already in production" (fit on train) ----
    base_model = LightGBMBaseline()
    base_model.fit(train, TARGET_COL)
    val_pred = base_model.predict(val)
    val_metrics = evaluate(val[TARGET_COL], val_pred, val["capacity_mw"])
    print(f"[trigger={args.trigger}] Base model validation MAE at training time: {val_metrics['MAE_MW']:.4f} MW")

    # ---- step 1: ingest "new observations" (here: the test split) ----
    new_obs = test.copy()
    print(f"1. Ingested {len(new_obs):,} new observation rows "
          f"({new_obs['timestamp'].min()} -> {new_obs['timestamp'].max()})")

    # ---- step 2: validate ----
    checked = run_all_checks(new_obs, production_col=TARGET_COL)
    summary = checked.attrs["physical_check_summary"]
    print(f"2. Validation: {summary['pct_flagged_anomalous']:.2f}% of new rows flagged anomalous")
    new_obs_valid = checked[~checked["physical_anomaly_flag"]]

    # ---- step 3: compute forecast error vs current model ----
    new_obs_valid = new_obs_valid.copy()
    new_obs_valid["raw_forecast_mw"] = base_model.predict(new_obs_valid)
    new_metrics = evaluate(new_obs_valid[TARGET_COL], new_obs_valid["raw_forecast_mw"], new_obs_valid["capacity_mw"])
    print(f"3. New-batch MAE with current model: {new_metrics['MAE_MW']:.4f} MW "
          f"(training-time val MAE: {val_metrics['MAE_MW']:.4f} MW)")

    # ---- step 4: update residual-correction model ----
    corrector = RollingBiasCorrector(window=10_000)
    corrector.fit(new_obs_valid, actual_col=TARGET_COL, forecast_col="raw_forecast_mw")
    new_obs_valid["corrected_forecast_mw"] = corrector.correct(new_obs_valid, forecast_col="raw_forecast_mw")
    corrected_metrics = evaluate(new_obs_valid[TARGET_COL], new_obs_valid["corrected_forecast_mw"], new_obs_valid["capacity_mw"])
    print(f"4. Residual correction updated. Corrected MAE: {corrected_metrics['MAE_MW']:.4f} MW")

    # ---- step 5: recalibrate uncertainty (conformal offsets) on new batch ----
    qmodel_feature_cols = [c for c in base_model.feature_cols]
    qmodel = QuantileForecastModel(feature_cols=qmodel_feature_cols)
    qmodel.fit(train, TARGET_COL)  # quantile models still need their own fit; recalibrate only
    qmodel.calibrate(new_obs_valid, TARGET_COL)
    q_pred = qmodel.predict(new_obs_valid)
    coverage = evaluate_coverage(new_obs_valid[TARGET_COL].reset_index(drop=True),
                                  q_pred["p10_mw"].reset_index(drop=True), q_pred["p90_mw"].reset_index(drop=True))
    print(f"5. Uncertainty recalibrated on new batch. Empirical 80% coverage: "
          f"{coverage['empirical_coverage_pct']:.1f}%")

    # ---- step 6: retrain decision ----
    degraded = new_metrics["MAE_MW"] > DEGRADATION_THRESHOLD * val_metrics["MAE_MW"]
    decision = "RETRAIN_TRIGGERED" if degraded else "NO_RETRAIN_NEEDED"
    print(f"6. Retrain decision: {decision} "
          f"(rule: new-batch MAE > {DEGRADATION_THRESHOLD}x training-time val MAE)")

    retrained_model_metrics = None
    if degraded:
        combined_train = pd.concat([train, new_obs_valid.drop(
            columns=["raw_forecast_mw", "corrected_forecast_mw"], errors="ignore")])
        new_model = LightGBMBaseline()
        new_model.fit(combined_train, TARGET_COL)
        retrained_pred = new_model.predict(val)
        retrained_model_metrics = evaluate(val[TARGET_COL], retrained_pred, val["capacity_mw"])
        # promotion rule: only replace if it doesn't regress validation MAE
        if retrained_model_metrics["MAE_MW"] <= val_metrics["MAE_MW"] * 1.05:
            base_model = new_model
            print(f"   -> New model PROMOTED (val MAE {retrained_model_metrics['MAE_MW']:.4f} "
                  f"<= 1.05x old {val_metrics['MAE_MW']:.4f})")
        else:
            print(f"   -> New model NOT promoted (val MAE {retrained_model_metrics['MAE_MW']:.4f} "
                  f"regressed beyond old {val_metrics['MAE_MW']:.4f})")

    # ---- step 7: save versioned artifacts ----
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = REGISTRY_DIR / version
    version_dir.mkdir(exist_ok=True)

    import joblib
    joblib.dump(base_model.model, version_dir / "lightgbm_model.joblib")
    joblib.dump(corrector, version_dir / "residual_corrector.joblib")

    metadata = {
        "version": version,
        "trigger": args.trigger,
        "data_mode": mode.value,
        "trained_on": TARGET_COL,
        "training_time_val_MAE_MW": val_metrics["MAE_MW"],
        "new_batch_MAE_MW_before_correction": new_metrics["MAE_MW"],
        "new_batch_MAE_MW_after_correction": corrected_metrics["MAE_MW"],
        "uncertainty_empirical_coverage_pct": coverage["empirical_coverage_pct"],
        "retrain_decision": decision,
        "retrained_model_val_MAE_MW": retrained_model_metrics["MAE_MW"] if retrained_model_metrics else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n7. Saved model version {version} -> {version_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
