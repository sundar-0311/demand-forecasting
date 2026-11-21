"""Tests focus on what silently breaks forecasting projects: leakage and train/serve skew."""

import numpy as np
import pandas as pd
import pytest

from demand.features import FEATURES, MIN_HISTORY, build_training_frame, features_at
from demand.forecast import recursive_forecast, seasonal_naive
from demand.data import SalesPanel
from demand.intervals import IntervalCalibrator
from demand.metrics import wape, smape, bias

N_DAYS, N_SERIES = 500, 4


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=N_DAYS, freq="D")
    values = rng.poisson(50, size=(N_DAYS, N_SERIES)).astype(float)
    return SalesPanel(dates, np.array([1, 1, 2, 2]), np.array([1, 2, 1, 2]), values)


def test_features_ignore_current_and_future_rows(panel):
    t = 450
    base = features_at(panel.values, t, panel.dates[t], panel.stores, panel.items)
    tampered = panel.values.copy()
    tampered[t:] = 1e9   # corrupt day t and everything after it
    after = features_at(tampered, t, panel.dates[t], panel.stores, panel.items)
    pd.testing.assert_frame_equal(base, after)


def test_feature_values_match_manual_calculation(panel):
    t = 400
    X = features_at(panel.values, t, panel.dates[t], panel.stores, panel.items)
    v = panel.values
    np.testing.assert_allclose(X["lag_7"], v[t - 7])
    np.testing.assert_allclose(X["lag_364"], v[t - 364])
    np.testing.assert_allclose(X["rolling_mean_7"], v[t - 7:t].mean(axis=0))
    np.testing.assert_allclose(X["rolling_std_28"], v[t - 28:t].std(axis=0, ddof=1))
    assert list(X.columns) == FEATURES


def test_training_frame_rows_match_features_at(panel):
    X, y, day_idx = build_training_frame(panel)
    assert len(X) == (N_DAYS - MIN_HISTORY) * N_SERIES
    t = 420
    rows = day_idx == t
    expected = features_at(panel.values, t, panel.dates[t], panel.stores, panel.items)
    pd.testing.assert_frame_equal(X[rows].reset_index(drop=True), expected)
    np.testing.assert_allclose(y[rows], panel.values[t])


class LagOneModel:
    """Predicts yesterday's sales: makes recursion easy to reason about."""
    def predict(self, X):
        return X["lag_1"].to_numpy()


def test_recursive_forecast_feeds_predictions_back(panel):
    history = panel.values[:450]
    out = recursive_forecast(LagOneModel(), history, panel.dates[449], 5, panel.stores, panel.items)
    # every step repeats the last observed value, because each prediction becomes the next lag_1
    for h in range(5):
        np.testing.assert_allclose(out[h], history[-1])


def test_recursive_forecast_never_sees_the_future(panel):
    origin = 450
    a = recursive_forecast(LagOneModel(), panel.values[:origin], panel.dates[origin - 1], 10,
                           panel.stores, panel.items)
    changed = panel.values.copy()
    changed[origin:] = 1e9
    b = recursive_forecast(LagOneModel(), changed[:origin], panel.dates[origin - 1], 10,
                           panel.stores, panel.items)
    np.testing.assert_allclose(a, b)


def test_seasonal_naive_uses_same_weekday(panel):
    history = panel.values[:450]
    out = seasonal_naive(history, horizon=14)
    np.testing.assert_allclose(out[0], history[450 - 7])
    np.testing.assert_allclose(out[6], history[450 - 1])
    np.testing.assert_allclose(out[7], history[450 - 7])    # second week still uses last observed week
    avg = seasonal_naive(history, horizon=1, weeks=2)
    np.testing.assert_allclose(avg[0], (history[450 - 7] + history[450 - 14]) / 2)


def test_metrics():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 30.0])
    assert wape(y, p) == pytest.approx(4 / 60 * 100)
    assert bias(y, p) == pytest.approx(0.0)
    assert smape(y, y) == 0.0


def test_interval_calibration_covers_and_widens_with_level():
    rng = np.random.default_rng(1)
    pred = rng.uniform(20, 100, size=(28, 2000))
    actual = pred + rng.normal(0, 1, pred.shape) * np.sqrt(pred)
    cal = IntervalCalibrator.fit([actual], [pred])
    fresh_actual = pred + rng.normal(0, 1, pred.shape) * np.sqrt(pred)
    cov80 = cal.coverage(fresh_actual, pred, 0.8)
    cov90 = cal.coverage(fresh_actual, pred, 0.9)
    assert 0.77 < cov80 < 0.83
    assert 0.87 < cov90 < 0.93
    lo80, hi80 = cal.bounds(pred, 0.8)
    lo90, hi90 = cal.bounds(pred, 0.9)
    assert np.all(hi90 >= hi80) and np.all(lo90 <= lo80) and np.all(lo80 >= 0)
