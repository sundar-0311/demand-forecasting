"""Prediction intervals via empirical residual quantiles (split-conformal style).

Backtest forecasts give us out-of-sample errors at every horizon step. Demand
noise grows with volume, so errors are standardised by sqrt(forecast) (a Poisson-like
scale). For each horizon step we store the quantiles of those standardised errors;
an interval is then `forecast + q * sqrt(forecast)`. No distributional assumption
is made, and uncertainty widens naturally as the horizon grows.
"""

import json
from pathlib import Path

import numpy as np

QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)
INTERVAL_LEVELS = (0.5, 0.8, 0.9)   # central coverage levels these quantiles support


def _scale(pred):
    return np.sqrt(np.maximum(pred, 1.0))


class IntervalCalibrator:
    def __init__(self, quantiles: np.ndarray):
        # quantiles[h, j]: standardised-error quantile QUANTILE_LEVELS[j] at horizon step h
        self.quantiles = np.asarray(quantiles)

    @property
    def horizon(self) -> int:
        return self.quantiles.shape[0]

    @classmethod
    def fit(cls, actuals: list[np.ndarray], preds: list[np.ndarray]) -> "IntervalCalibrator":
        """`actuals`/`preds`: one (horizon, n_series) array per backtest fold."""
        z = np.concatenate([(a - p) / _scale(p) for a, p in zip(actuals, preds)], axis=1)
        return cls(np.quantile(z, QUANTILE_LEVELS, axis=1).T)

    def bounds(self, pred: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper bounds for a (horizon,) or (horizon, n_series) forecast."""
        if level not in INTERVAL_LEVELS:
            raise ValueError(f"level must be one of {INTERVAL_LEVELS}")
        lo_q = QUANTILE_LEVELS.index(round((1 - level) / 2, 2))
        hi_q = QUANTILE_LEVELS.index(round((1 + level) / 2, 2))
        h = pred.shape[0]
        q = self.quantiles[:h]
        if pred.ndim == 2:
            q = q[:, :, None]
        lower = np.clip(pred + q[:, lo_q] * _scale(pred), 0, None)
        upper = pred + q[:, hi_q] * _scale(pred)
        return lower, upper

    def upper_quantile(self, pred: np.ndarray, quantile: float) -> np.ndarray:
        """Upper bound at a one-sided quantile (0.75, 0.9 or 0.95): used for stock levels."""
        j = QUANTILE_LEVELS.index(round(quantile, 2))
        return pred + self.quantiles[:pred.shape[0], j] * _scale(pred)

    def coverage(self, actual: np.ndarray, pred: np.ndarray, level: float) -> float:
        lower, upper = self.bounds(pred, level)
        return float(np.mean((actual >= lower) & (actual <= upper)))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "quantile_levels": QUANTILE_LEVELS,
            "quantiles": self.quantiles.tolist(),
        }))

    @classmethod
    def load(cls, path: str | Path) -> "IntervalCalibrator":
        return cls(np.array(json.loads(Path(path).read_text())["quantiles"]))
