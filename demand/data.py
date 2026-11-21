"""Loading sales data into a dense (days x series) panel."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SalesPanel:
    """Daily unit sales for every (store, item) series on a shared calendar."""

    dates: pd.DatetimeIndex   # (T,)
    stores: np.ndarray        # (S,) store id of each column
    items: np.ndarray         # (S,) item id of each column
    values: np.ndarray        # (T, S) units sold

    @property
    def n_days(self) -> int:
        return self.values.shape[0]

    @property
    def n_series(self) -> int:
        return self.values.shape[1]

    def column(self, store: int, item: int) -> int:
        hits = np.flatnonzero((self.stores == store) & (self.items == item))
        if len(hits) != 1:
            raise KeyError(f"No series for store={store}, item={item}")
        return int(hits[0])

    def to_long(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": np.repeat(self.dates.values, self.n_series),
            "store": np.tile(self.stores, self.n_days),
            "item": np.tile(self.items, self.n_days),
            "sales": self.values.ravel().astype(int),
        })


def load_panel(path: str | Path) -> SalesPanel:
    """Read a long-format CSV (date, store, item, sales) into a SalesPanel.

    Every series must have a value on every day; the Kaggle store-item data does.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    wide = df.pivot(index="date", columns=["store", "item"], values="sales").sort_index()
    if wide.isna().any().any():
        raise ValueError("Panel has missing store/item/day combinations; fill or drop them first.")
    stores = np.array([c[0] for c in wide.columns])
    items = np.array([c[1] for c in wide.columns])
    return SalesPanel(wide.index, stores, items, wide.to_numpy(dtype=float))
