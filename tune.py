"""Hyper-parameter search for LightGBM and XGBoost with time-ordered cross-validation.

Only data from BEFORE the first backtest origin is used, so the backtest that
follows is a clean out-of-sample test of the tuned settings.

Rows are ordered by day, so `TimeSeriesSplit` really does split by time: each
validation block comes strictly after its training block, with a 28-day gap
(the forecast horizon) so lag/rolling features cannot straddle the boundary.
"""

import json
import time

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from demand.config import RAW_DATA_PATH, REPORTS, SEED, HORIZON, backtest_origins
from demand.data import load_panel
from demand.features import build_training_frame
from demand.models import SEARCH_SPACES, build_model

N_ITER = 12
N_SPLITS = 4
VALIDATION_DAYS = 56


def main():
    REPORTS.mkdir(exist_ok=True)
    panel = load_panel(RAW_DATA_PATH)
    X, y, day_idx = build_training_frame(panel)

    first_origin = backtest_origins(panel.n_days)[0]
    keep = day_idx < first_origin
    X, y = X[keep].reset_index(drop=True), y[keep]
    print(f"Tuning on {len(X):,} rows (days before {panel.dates[first_origin].date()})")

    n = panel.n_series
    cv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=n * VALIDATION_DAYS, gap=n * HORIZON)

    best = {}
    for name, space in SEARCH_SPACES.items():
        start = time.time()
        search = RandomizedSearchCV(
            build_model(name), {k: v for k, v in space.items()},
            n_iter=N_ITER, cv=cv, scoring="neg_root_mean_squared_error",
            random_state=SEED, n_jobs=1, verbose=1,
        )
        search.fit(X, y)
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
        best[name] = {"params": params, "cv_rmse": float(-search.best_score_)}
        print(f"\n{name}: CV RMSE {-search.best_score_:.3f} in {time.time() - start:.0f}s -> {params}")

    (REPORTS / "best_params.json").write_text(json.dumps(best, indent=2))
    print(f"Saved {REPORTS / 'best_params.json'}")


if __name__ == "__main__":
    main()
