"""
Actually fits and backtests ONE MODEL PER HORIZON (intra-day, J+1, J+2, J+3) -
CDC continuation Phase 5/6. Writes reports/MULTI_HORIZON_REPORT.md's
per-horizon section.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.data_mode import get_data_mode, DataMode
from src.features.lag_features import add_lag_features, add_rolling_features
from src.forecasting.backtesting import make_chrono_split, evaluate, evaluate_by_group
from src.forecasting.multi_horizon import HorizonModel, HORIZONS
from src.uncertainty.quantile_forecast import QuantileForecastModel, evaluate_coverage, pinball_loss
from src.validation.physical_checks import run_all_checks
from src.validation.physical_constraints import apply_physical_constraints, measure_constraint_impact

# Prioritize reference dataset
DATA_PATH = Path("data/processed/training_dataset_15min_real_reference.csv")
TARGET_COL = "pv_production_mw_reference"

if not DATA_PATH.exists():
    DATA_PATH = Path("data/processed/training_dataset_15min.csv")
    TARGET_COL = "pv_production_mw_proxy"


def main():
    if not DATA_PATH.exists():
        print(f"Dataset {DATA_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    
    # We already have lag features in the real reference training dataset (Script 10),
    # but if we are on the proxy fallback dataset, we might need to compute them.
    if f"{TARGET_COL}_lag_1" not in df.columns:
        df = add_lag_features(df, TARGET_COL)
        df = add_rolling_features(df, TARGET_COL)

    split = make_chrono_split(df, train_frac=0.6, val_frac=0.2)
    train, val, test = split.split(df)

    all_point_results = {}
    all_constraint_impacts = {}
    all_uncertainty_results = {}
    all_group_results = {}

    for horizon_name in HORIZONS:
        print(f"\n=== Horizon: {horizon_name} ===")
        feats = HORIZONS[horizon_name]["features"]
        # Ensure we only use feature columns that actually exist in the dataframe
        feats = [f for f in feats if f in df.columns]
        
        hm = HorizonModel(horizon_name=horizon_name, feature_cols=list(feats))

        train_h = hm.build_supervised_frame(train, TARGET_COL)
        val_h = hm.build_supervised_frame(val, TARGET_COL)
        test_h = hm.build_supervised_frame(test, TARGET_COL)

        hm.fit(train_h)
        pred_h = hm.predict(test_h)

        # Apply physical constraints
        test_h["raw_forecast_mw"] = pred_h
        test_h["constrained_forecast_mw"] = apply_physical_constraints(
            test_h, "raw_forecast_mw", "capacity_mw", "solar_elevation_deg"
        )
        
        valid_rows = test_h["target_at_horizon"].notna() & test_h["constrained_forecast_mw"].notna()
        m = evaluate(test_h.loc[valid_rows, "target_at_horizon"], test_h.loc[valid_rows, "constrained_forecast_mw"],
                     test_h.loc[valid_rows, "capacity_mw"])
        all_point_results[horizon_name] = m
        print("Point forecast metrics (after constraints):", m)
        
        impact = measure_constraint_impact(test_h, test_h["raw_forecast_mw"], test_h["constrained_forecast_mw"])
        all_constraint_impacts[horizon_name] = impact
        print("Constraints impact:", impact)

        # physical validation of the forecast itself (Phase 18)
        fc_df = test_h.loc[valid_rows, ["capacity_mw", "daylight_flag"]].copy()
        fc_df["forecast_mw"] = test_h.loc[valid_rows, "constrained_forecast_mw"]
        checked = run_all_checks(fc_df, production_col="forecast_mw")
        print("Forecast physical-check summary:", checked.attrs["physical_check_summary"])

        # uncertainty per horizon
        qfeat = [c for c in feats if c in train_h.columns]
        qm = QuantileForecastModel(feature_cols=qfeat)
        qm.fit(train_h.rename(columns={"target_at_horizon": "target_at_horizon"}), "target_at_horizon")
        qm.calibrate(val_h, "target_at_horizon")
        q_test = qm.predict(test_h)
        
        # Apply physical constraints to quantiles as well
        for q_col in ["p10_mw", "p25_mw", "p50_mw", "p75_mw", "p90_mw"]:
            q_test[q_col] = apply_physical_constraints(
                pd.concat([test_h[["capacity_mw", "solar_elevation_deg"]], q_test[[q_col]]], axis=1), 
                q_col, "capacity_mw", "solar_elevation_deg"
            )
            
        cov = evaluate_coverage(test_h["target_at_horizon"].reset_index(drop=True),
                                 q_test["p10_mw"].reset_index(drop=True),
                                 q_test["p90_mw"].reset_index(drop=True))
        pb = {f"pinball_{t}": pinball_loss(test_h["target_at_horizon"].to_numpy(), q_test[c].to_numpy(), t)
              for t, c in zip([0.10, 0.25, 0.50, 0.75, 0.90],
                               ["p10_mw", "p25_mw", "p50_mw", "p75_mw", "p90_mw"])}
        all_uncertainty_results[horizon_name] = {**cov, **pb}
        print("Uncertainty (after constraints):", all_uncertainty_results[horizon_name])

        # per-governorate breakdown for this horizon
        gdf = test_h.loc[valid_rows, ["governorate", "capacity_mw"]].copy()
        gdf["y_true"] = test_h.loc[valid_rows, "target_at_horizon"].to_numpy()
        gdf["y_pred"] = test_h.loc[valid_rows, "constrained_forecast_mw"].to_numpy()
        gres = evaluate_by_group(gdf, "y_true", "y_pred", "capacity_mw", ["governorate"])
        all_group_results[horizon_name] = gres

    Path("reports").mkdir(exist_ok=True)
    with open("reports/MULTI_HORIZON_REPORT.md", "w") as f:
        f.write("# MULTI_HORIZON_REPORT.md\n\n")
        f.write(f"**Target:** `{TARGET_COL}`\n")
        f.write("One LightGBM model was actually fit and backtested PER HORIZON (not a single model reused across "
                "horizons) - see `src/forecasting/multi_horizon.py` for the feature-availability rationale.\n\n")
        
        f.write("## Physical Constraint Impact\n\n")
        f.write("Forecasts were post-processed to enforce P_pv=0 at night and P_pv <= Capacity.\n\n")
        f.write(pd.DataFrame(all_constraint_impacts).T.round(3).to_markdown() + "\n\n")

        f.write("## Point-forecast metrics by horizon (national-level test set, after constraints)\n\n")
        f.write(pd.DataFrame(all_point_results).T.round(3).to_markdown() + "\n\n")
        
        f.write("## Uncertainty by horizon (80% interval coverage, pinball losses, after constraints)\n\n")
        f.write(pd.DataFrame(all_uncertainty_results).T.round(4).to_markdown() + "\n\n")
        
        f.write("## Per-governorate metrics, by horizon (after constraints)\n\n")
        for h, gres in all_group_results.items():
            f.write(f"### {h}\n\n")
            f.write(gres.round(3).to_markdown(index=False) + "\n\n")

    print("\nWrote reports/MULTI_HORIZON_REPORT.md")


if __name__ == "__main__":
    main()

