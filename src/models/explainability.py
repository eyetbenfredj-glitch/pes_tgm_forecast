"""
Explainability (CDC continuation Phase 13). Wires SHAP TreeExplainer to the
already-fitted LightGBM model, producing:
  - global feature importance (mean |SHAP value| per feature)
  - one local explanation (single row: district + timestamp)
Saved under reports/explainability/ as CSV + a short markdown summary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def global_feature_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance


def local_explanation(model, X: pd.DataFrame, row_index) -> pd.DataFrame:
    import shap
    explainer = shap.TreeExplainer(model)
    row = X.loc[[row_index]]
    shap_values = explainer.shap_values(row)
    base_value = explainer.expected_value
    out = pd.DataFrame({
        "feature": X.columns,
        "feature_value": row.iloc[0].to_numpy(),
        "shap_value": shap_values[0],
    }).sort_values("shap_value", key=np.abs, ascending=False).reset_index(drop=True)
    out.attrs["base_value"] = float(base_value)
    out.attrs["prediction"] = float(base_value + shap_values[0].sum())
    return out


def run_and_save(model, X: pd.DataFrame, example_row_index, meta: dict,
                  out_dir: str = "reports/explainability") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    gfi = global_feature_importance(model, X)
    gfi.to_csv(f"{out_dir}/global_feature_importance.csv", index=False)

    local = local_explanation(model, X, example_row_index)
    local.to_csv(f"{out_dir}/local_explanation_example.csv", index=False)

    with open(f"{out_dir}/EXPLAINABILITY_SUMMARY.md", "w") as f:
        f.write("# Explainability summary (SHAP, LightGBM)\n\n")
        f.write(f"Model: {meta.get('model_name', 'unknown')} | Trained on: {meta.get('trained_on', 'unknown')}\n\n")
        f.write("## Global feature importance (mean |SHAP value|)\n\n")
        f.write(gfi.head(10).to_markdown(index=False) + "\n\n")
        f.write(f"## Local explanation example (district={meta.get('district', '?')}, "
                f"timestamp={meta.get('timestamp', '?')})\n\n")
        f.write(f"Base value: {local.attrs['base_value']:.4f} MW | "
                f"Prediction: {local.attrs['prediction']:.4f} MW\n\n")
        f.write(local.head(10).to_markdown(index=False) + "\n")

    print(f"Wrote {out_dir}/global_feature_importance.csv, "
          f"{out_dir}/local_explanation_example.csv, {out_dir}/EXPLAINABILITY_SUMMARY.md")
