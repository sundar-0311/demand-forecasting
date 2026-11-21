"""Shared experiment settings, so every script evaluates the same way."""

from pathlib import Path

RAW_DATA_PATH = Path("data/real_sales_data.csv")   # Kaggle train.csv (date, store, item, sales)
HISTORY_PATH = Path("app_data/history.csv.gz")     # compact slice the deployed app reads
MODEL_PATH = Path("models/demand_model.pkl")
META_PATH = Path("models/meta.json")
REPORTS = Path("reports")

HORIZON = 28          # days forecast recursively from each origin
N_FOLDS = 6           # rolling-origin backtest folds
FOLD_STEP = 55        # days between origins; NOT a multiple of 7, so folds start on different weekdays
                      # (otherwise "days ahead" would be confounded with day of week)
HISTORY_DAYS = 400    # days of history the app needs (>= 364-day lag + a margin)
SEED = 42


def backtest_origins(n_days: int) -> list[int]:
    """Row indices of the forecast origins, oldest first.

    The newest origin forecasts the last HORIZON days of the data, so every
    forecast in the backtest is scored against actuals the model never saw.
    """
    latest = n_days - HORIZON
    return [latest - FOLD_STEP * k for k in reversed(range(N_FOLDS))]
