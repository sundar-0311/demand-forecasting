"""Forecast accuracy metrics. All take arrays of any matching shape."""

import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wape(y, p):
    """Weighted absolute % error: total absolute error / total actual. Robust to small values."""
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100)


def smape(y, p):
    denom = (np.abs(y) + np.abs(p)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(y - p)[mask] / denom[mask]) * 100)


def bias(y, p):
    """Mean signed error as % of mean actual. Positive = over-forecasting."""
    return float(np.sum(p - y) / np.sum(y) * 100)


def summarize(y, p) -> dict:
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "WAPE": wape(y, p), "sMAPE": smape(y, p), "Bias%": bias(y, p)}
