"""Fit the production model on ALL data and export what the app needs.

Run after tune.py and backtest.py (backtest picks the best model and calibrates
the prediction intervals). Writes:
  models/demand_model.pkl      the fitted model
  models/meta.json             model name, last data date, feature list
  app_data/history.csv.gz      the most recent HISTORY_DAYS of sales (the app's only data source)
  reports/shap_*.png           SHAP explanations of the model
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from demand.config import HISTORY_DAYS, HISTORY_PATH, META_PATH, MODEL_PATH, RAW_DATA_PATH, REPORTS, SEED
from demand.data import load_panel
from demand.features import FEATURES, build_training_frame
from demand.models import build_model


def main():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    HISTORY_PATH.parent.mkdir(exist_ok=True)

    panel = load_panel(RAW_DATA_PATH)
    X, y, _ = build_training_frame(panel)

    best_name = json.loads((REPORTS / "best_model.json").read_text())["model"]
    tuned = json.loads((REPORTS / "best_params.json").read_text()) if (REPORTS / "best_params.json").exists() else {}
    params = tuned.get(best_name, {}).get("params")
    print(f"Fitting {best_name} on {len(X):,} rows (through {panel.dates[-1].date()})")
    model = build_model(best_name, params)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH, compress=3)
    META_PATH.write_text(json.dumps({
        "model": best_name,
        "params": params,
        "last_date": str(panel.dates[-1].date()),
        "features": FEATURES,
    }, indent=2))

    long = panel.to_long()
    long[long["date"] >= panel.dates[-HISTORY_DAYS]].to_csv(HISTORY_PATH, index=False)
    print(f"Saved {MODEL_PATH} and {HISTORY_PATH}")

    explain(model, X, best_name)


def explain(model, X, name):
    """SHAP values on a sample of the most recent year (tree models only)."""
    if name not in ("LightGBM", "XGBoost", "Random Forest"):
        return
    sample = X.iloc[-365 * 500:].sample(4000, random_state=SEED)
    shap_values = shap.TreeExplainer(model).shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=15)
    plt.title("Mean |SHAP|: average impact on forecast (units)")
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_importance.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=150)
    plt.close()

    importance = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(sample.columns, importance), key=lambda kv: -kv[1])
    print("Top drivers (mean |SHAP|):", ", ".join(f"{n} {v:.1f}" for n, v in ranked[:6]))
    (REPORTS / "shap_importance.json").write_text(json.dumps({n: float(v) for n, v in ranked}, indent=2))


if __name__ == "__main__":
    main()
