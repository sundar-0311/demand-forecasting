"""Feature engineering.

One function, `features_at`, builds the features for a single forecast day from
the sales history that precedes it. Training and recursive forecasting both use
it, so there is no train/serve skew, and nothing at or after day `t` can leak in.
"""

import numpy as np
import pandas as pd

from .data import SalesPanel

LAGS = (1, 7, 14, 28, 364)
WINDOWS = (7, 14, 28)
MIN_HISTORY = max(LAGS)   # first day that has every lag available

CALENDAR_FEATURES = ["day_of_week", "is_weekend", "day_of_month", "week_of_year", "month", "quarter"]
FEATURES = (
    ["store", "item"]
    + CALENDAR_FEATURES
    + [f"lag_{k}" for k in LAGS]
    + [f"rolling_mean_{w}" for w in WINDOWS]
    + [f"rolling_std_{w}" for w in WINDOWS]
)


def features_at(values: np.ndarray, t: int, date: pd.Timestamp,
                stores: np.ndarray, items: np.ndarray) -> pd.DataFrame:
    """Features for day `t` of every series, using only rows `values[:t]`.

    `values` may contain anything from row `t` onward (zeros, forecasts, actuals):
    those rows are never read.
    """
    if t < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} days of history, got {t}")
    n_series = values.shape[1]
    dow = date.dayofweek
    iso_week = date.isocalendar().week

    data = {
        "store": stores,
        "item": items,
        "day_of_week": np.full(n_series, dow),
        "is_weekend": np.full(n_series, int(dow >= 5)),
        "day_of_month": np.full(n_series, date.day),
        "week_of_year": np.full(n_series, int(iso_week)),
        "month": np.full(n_series, date.month),
        "quarter": np.full(n_series, date.quarter),
    }
    for k in LAGS:
        data[f"lag_{k}"] = values[t - k]
    for w in WINDOWS:
        window = values[t - w:t]
        data[f"rolling_mean_{w}"] = window.mean(axis=0)
        data[f"rolling_std_{w}"] = window.std(axis=0, ddof=1)
    return pd.DataFrame(data)[FEATURES]


def build_training_frame(panel: SalesPanel):
    """Features, target and day index for every (day, series) with enough history.

    Rows are ordered by day, so time-based splits are simple slices of `day_idx`.
    """
    frames, targets, day_idx = [], [], []
    for t in range(MIN_HISTORY, panel.n_days):
        frames.append(features_at(panel.values, t, panel.dates[t], panel.stores, panel.items))
        targets.append(panel.values[t])
        day_idx.append(np.full(panel.n_series, t))
    return pd.concat(frames, ignore_index=True), np.concatenate(targets), np.concatenate(day_idx)
