"""Recursive multi-step forecasting and simple seasonal baselines."""

import numpy as np
import pandas as pd

from .features import features_at


def recursive_forecast(model, history: np.ndarray, last_date: pd.Timestamp, horizon: int,
                       stores: np.ndarray, items: np.ndarray) -> np.ndarray:
    """Forecast `horizon` days ahead of `history`, feeding each prediction back in.

    A one-step model only knows yesterday's sales. To look 28 days out, day h's
    prediction becomes the "actual" that day h+1's lags and rolling stats read.
    Returns an array of shape (horizon, n_series).
    """
    n_hist, n_series = history.shape
    buffer = np.vstack([history, np.zeros((horizon, n_series))])
    future = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    out = np.empty((horizon, n_series))
    for h in range(horizon):
        X = features_at(buffer, n_hist + h, future[h], stores, items)
        pred = np.clip(model.predict(X), 0, None)
        buffer[n_hist + h] = pred
        out[h] = pred
    return out


def seasonal_naive(history: np.ndarray, horizon: int, weeks: int = 1) -> np.ndarray:
    """Mean of the same weekday over the last `weeks` weeks (weeks=1 is plain seasonal naive)."""
    n_hist = history.shape[0]
    out = np.empty((horizon, history.shape[1]))
    for h in range(horizon):
        t = n_hist + h
        base = t - 7 * (h // 7 + 1)   # most recent observed day with the same weekday
        out[h] = np.mean([history[base - 7 * j] for j in range(weeks)], axis=0)
    return out
