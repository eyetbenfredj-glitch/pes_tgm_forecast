"""
Script 08: XGBoost Time-Aware Hyperparameter Tuning.

Performs hyperparameter tuning for the XGBoost model using Optuna.
Uses chronological validation (no random k-fold) to respect time series structure.

Search space:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight
- reg_alpha
- reg_lambda
- gamma

Outputs:
- reports/xgboost_best_params.json
- reports/xgboost_tuning_trials.csv

Usage:
    python scripts/08_xgboost_tuning.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.backtesting import make_chrono_split
from src.models.baselines import DEFAULT_ML_FEATURES

TARGET_COL = "pv_production_mw_reference"
DATA_PATH = Path("data/processed/training_dataset_15min_real_reference.csv")
OUT_JSON = Path("reports/xgboost_best_params.json")
OUT_CSV = Path("reports/xgboost_tuning_trials.csv")


def main():
    print("=" * 60)
    print("Script 08: XGBoost Time-Aware Hyperparameter Tuning")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run script 10 first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    # Fallback to proxy if reference target not found
    if TARGET_COL not in df.columns:
        print(f"Target '{TARGET_COL}' not found. Falling back to 'pv_production_mw_proxy'.")
        target = "pv_production_mw_proxy"
    else:
        target = TARGET_COL

    features = [c for c in DEFAULT_ML_FEATURES if c in df.columns]

    print("Chronological split (train/val) ...")
    split = make_chrono_split(df, train_frac=0.7, val_frac=0.3)
    train = df[df["timestamp"] <= split.train_end]
    val = df[(df["timestamp"] > split.train_end) & (df["timestamp"] <= split.val_end)]

    X_train = train[features]
    y_train = train[target]
    X_val = val[features]
    y_val = val[target]

    # Filter out NaNs for training
    valid_train = y_train.notna() & X_train.notna().all(axis=1)
    X_train, y_train = X_train[valid_train], y_train[valid_train]

    valid_val = y_val.notna() & X_val.notna().all(axis=1)
    X_val, y_val = X_val[valid_val], y_val[valid_val]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)
        return rmse

    print("Starting Optuna optimization (20 trials) ...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("\nBest parameters:")
    best_params = study.best_params
    print(json.dumps(best_params, indent=2))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best params to {OUT_JSON}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUT_CSV, index=False)
    print(f"Saved tuning trials to {OUT_CSV}")


if __name__ == "__main__":
    main()
