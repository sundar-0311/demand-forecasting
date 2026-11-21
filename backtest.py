"""Rolling-origin backtest of recursive 28-day forecasts.

For each of N_FOLDS forecast origins (spaced FOLD_STEP days apart) every model is
re-trained on data before the origin only, then asked to forecast the next 28 days
recursively, with no access to actuals after the origin. This mirrors real use and
exposes error growth with horizon, which a single one-step holdout hides.

Outputs (reports/): backtest_metrics.csv, backtest_folds.csv, backtest_by_horizon.csv,
backtest_by_store.csv, backtest_by_item.csv, interval_coverage.csv,
interval_calibration.json, backtest_last_fold.csv, best_model.json and PNG charts.
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from demand.config import HORIZON, RAW_DATA_PATH, REPORTS, backtest_origins
from demand.data import load_panel
from demand.features import build_training_frame
from demand.forecast import recursive_forecast, seasonal_naive
from demand.intervals import IntervalCalibrator
from demand.metrics import summarize, wape
from demand.models import ML_MODELS, build_model

BASELINES = {
    "Seasonal naive (last week)": lambda hist: seasonal_naive(hist, HORIZON, weeks=1),
    "Seasonal avg (last 4 weeks)": lambda hist: seasonal_naive(hist, HORIZON, weeks=4),
}


def load_tuned_params() -> dict:
    path = REPORTS / "best_params.json"
    if not path.exists():
        print("No reports/best_params.json found: using default hyper-parameters (run tune.py first).")
        return {}
    return {name: v["params"] for name, v in json.loads(path.read_text()).items()}


def main():
    REPORTS.mkdir(exist_ok=True)
    panel = load_panel(RAW_DATA_PATH)
    X, y, day_idx = build_training_frame(panel)
    origins = backtest_origins(panel.n_days)
    tuned = load_tuned_params()

    forecasts = {}   # model -> list of (HORIZON, S) arrays, one per fold
    actuals = []
    for fold, origin in enumerate(origins):
        history = panel.values[:origin]
        actual = panel.values[origin:origin + HORIZON]
        actuals.append(actual)
        print(f"\nFold {fold + 1}/{len(origins)}: origin {panel.dates[origin].date()}")

        for name, fn in BASELINES.items():
            forecasts.setdefault(name, []).append(fn(history))

        train = day_idx < origin
        for name in ML_MODELS:
            start = time.time()
            model = build_model(name, tuned.get(name))
            model.fit(X[train], y[train])
            pred = recursive_forecast(model, history, panel.dates[origin - 1], HORIZON,
                                      panel.stores, panel.items)
            forecasts.setdefault(name, []).append(pred)
            print(f"  {name:<18} WAPE {wape(actual, pred):5.2f}%  ({time.time() - start:.0f}s)")

    # ---- aggregate metrics -------------------------------------------------
    fold_rows, summary_rows = [], []
    for name, preds in forecasts.items():
        for fold, (a, p) in enumerate(zip(actuals, preds)):
            fold_rows.append({"model": name, "fold": fold + 1, "origin": panel.dates[origins[fold]].date(),
                              **summarize(a, p)})
        summary_rows.append({"model": name, **summarize(np.concatenate(actuals), np.concatenate(preds))})
    summary = pd.DataFrame(summary_rows).set_index("model").sort_values("WAPE")
    base = summary.loc["Seasonal naive (last week)"]
    summary["MAE vs seasonal naive %"] = (summary["MAE"] / base["MAE"] - 1) * 100
    summary.to_csv(REPORTS / "backtest_metrics.csv")
    pd.DataFrame(fold_rows).to_csv(REPORTS / "backtest_folds.csv", index=False)
    print("\n=== Backtest (6 folds x 28-day recursive forecasts) ===")
    print(summary.round(2).to_string())

    best = summary.loc[[m for m in summary.index if m in ML_MODELS]].index[0]
    print(f"\nBest ML model by WAPE: {best}")
    (REPORTS / "best_model.json").write_text(json.dumps({"model": best}))

    # ---- error by horizon --------------------------------------------------
    rows = []
    for name, preds in forecasts.items():
        a, p = np.stack(actuals), np.stack(preds)    # (folds, H, S)
        for h in range(HORIZON):
            rows.append({"model": name, "horizon": h + 1, "WAPE": wape(a[:, h], p[:, h])})
    by_horizon = pd.DataFrame(rows)
    by_horizon.to_csv(REPORTS / "backtest_by_horizon.csv", index=False)

    # ---- error by store / item for the best model --------------------------
    a, p = np.concatenate(actuals), np.concatenate(forecasts[best])   # (folds*H, S)
    per_series = pd.DataFrame({"store": panel.stores, "item": panel.items,
                               "WAPE": [wape(a[:, s], p[:, s]) for s in range(panel.n_series)],
                               "avg_daily_units": a.mean(axis=0)})
    per_series.groupby("store").apply(
        lambda g: pd.Series({"WAPE": wape(a[:, g.index], p[:, g.index]), "avg_daily_units": g["avg_daily_units"].mean()}),
        include_groups=False).reset_index().to_csv(REPORTS / "backtest_by_store.csv", index=False)
    per_series.to_csv(REPORTS / "backtest_by_item.csv", index=False)

    # ---- prediction intervals ---------------------------------------------
    # Honest coverage check: calibrate on the older folds, test on the newest one.
    cal_check = IntervalCalibrator.fit(actuals[:-1], forecasts[best][:-1])
    coverage = pd.DataFrame([
        {"nominal": level, "empirical_coverage": cal_check.coverage(actuals[-1], forecasts[best][-1], level)}
        for level in (0.5, 0.8, 0.9)
    ])
    coverage.to_csv(REPORTS / "interval_coverage.csv", index=False)
    print("\nInterval coverage on the newest fold (calibrated on older folds only):")
    print(coverage.round(3).to_string(index=False))

    # Production calibration uses every fold.
    calibrator = IntervalCalibrator.fit(actuals, forecasts[best])
    calibrator.save(REPORTS / "interval_calibration.json")

    # ---- artefacts the app and README use ----------------------------------
    last_origin = origins[-1]
    last_dates = panel.dates[last_origin:last_origin + HORIZON]
    pd.DataFrame({
        "date": np.repeat(last_dates.values, panel.n_series),
        "store": np.tile(panel.stores, HORIZON),
        "item": np.tile(panel.items, HORIZON),
        "actual": actuals[-1].ravel(),
        "forecast": forecasts[best][-1].ravel(),
    }).to_csv(REPORTS / "backtest_last_fold.csv", index=False)

    make_charts(summary, by_horizon, best, panel, actuals[-1], forecasts[best][-1], calibrator, last_dates)


def make_charts(summary, by_horizon, best, panel, actual, pred, calibrator, last_dates):
    colors = {"Seasonal naive (last week)": "#9AA5B1", "Seasonal avg (last 4 weeks)": "#C4B5A0",
              "Linear Regression": "#D98C5F", "Random Forest": "#6BA38C", "XGBoost": "#5B8DB8",
              "LightGBM": "#1F5FA8"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, g in by_horizon.groupby("model"):
        lw = 2.8 if name == best else 1.4
        ax.plot(g["horizon"], g["WAPE"], label=name, color=colors.get(name), lw=lw)
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("WAPE (%)  lower is better")
    ax.set_title("Forecast error vs horizon (6-fold rolling-origin backtest)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "error_by_horizon.png", dpi=150)
    plt.close(fig)

    # Example forecast with interval: pick a mid-volume series so the band is visible.
    s = int(np.argsort(actual.mean(axis=0))[len(actual.T) // 2])
    lower, upper = calibrator.bounds(pred[:, s], 0.8)
    history_days = 42
    origin_idx = panel.n_days - HORIZON
    hist_dates = panel.dates[origin_idx - history_days:origin_idx]
    hist_vals = panel.values[origin_idx - history_days:origin_idx, s]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hist_dates, hist_vals, color="#9AA5B1", lw=1.5, label="History")
    ax.fill_between(last_dates, lower, upper, color="#1F5FA8", alpha=0.18, label="80% interval")
    ax.plot(last_dates, actual[:, s], color="#222", lw=1.5, marker="o", ms=3, label="Actual")
    ax.plot(last_dates, pred[:, s], color="#1F5FA8", lw=2, label=f"{best} forecast")
    ax.set_title(f"28-day recursive forecast, store {panel.stores[s]} / item {panel.items[s]} (held out)")
    ax.set_ylabel("Units sold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(REPORTS / "forecast_example.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    order = summary.sort_values("WAPE", ascending=False)
    ax.barh(order.index, order["WAPE"], color=[colors.get(m) for m in order.index])
    for i, v in enumerate(order["WAPE"]):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("WAPE (%)  lower is better")
    ax.set_title("Model comparison, 28-day recursive forecasts")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "model_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
