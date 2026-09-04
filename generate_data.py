"""
Data loader for the demand forecasting pipeline.

Two modes:
1. REAL DATA (preferred): if data/real_sales_data.csv exists, it's loaded and
   standardized. Expected format: Kaggle's "Store Item Demand Forecasting
   Challenge" dataset (https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)
   with columns: date, store, item, sales
   -> Download train.csv from Kaggle, rename it to real_sales_data.csv,
      and place it in the data/ folder.

2. SYNTHETIC DATA (fallback): if no real dataset is found, generates a
   realistic synthetic dataset with trend, seasonality, and promo spikes,
   so the pipeline always runs end-to-end even without Kaggle access.

Either way, the output is standardized to data/sales_data.csv with columns:
date, store, category, day_of_week, month, is_weekend, promo, units_sold
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
Path("data").mkdir(exist_ok=True)

REAL_DATA_PATH = Path("data/real_sales_data.csv")
OUTPUT_PATH = Path("data/sales_data.csv")


def load_real_data(path: Path) -> pd.DataFrame:
    """Load and standardize the Kaggle Store-Item Demand Forecasting dataset."""
    print(f"Found real dataset at {path} — loading and standardizing...")
    df = pd.read_csv(path, parse_dates=["date"])

    # Kaggle columns: date, store, item, sales -> standardize naming
    df = df.rename(columns={"item": "category", "sales": "units_sold"})
    df["store"] = "Store_" + df["store"].astype(str)
    df["category"] = "Item_" + df["category"].astype(str)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # No promo flag in this dataset — default to 0 (pipeline still works;
    # the promo feature simply won't carry predictive signal here)
    df["promo"] = 0

    cols = ["date", "store", "category", "day_of_week", "month", "is_weekend", "promo", "units_sold"]
    return df[cols].sort_values(["store", "category", "date"]).reset_index(drop=True)


def generate_synthetic_data() -> pd.DataFrame:
    """Generate a synthetic but realistic daily retail demand dataset."""
    print("No real dataset found — generating synthetic data instead.")
    print("(To use real data, download Kaggle's Store Item Demand Forecasting "
          "dataset, rename train.csv to real_sales_data.csv, and place it in data/)")

    START_DATE, END_DATE = "2022-01-01", "2024-12-31"
    STORES = ["Store_A", "Store_B", "Store_C"]
    CATEGORIES = ["Electronics", "Groceries", "Apparel"]
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    rows = []
    for store in STORES:
        store_base = np.random.uniform(80, 150)
        for category in CATEGORIES:
            cat_multiplier = {"Electronics": 0.7, "Groceries": 1.6, "Apparel": 1.0}[category]
            for i, date in enumerate(dates):
                trend = 0.02 * i
                day_of_year = date.dayofyear
                yearly_season = (
                    25 * np.sin(2 * np.pi * (day_of_year - 250) / 365)
                    if day_of_year > 200
                    else 5 * np.sin(2 * np.pi * day_of_year / 365)
                )
                weekly_season = 20 if date.dayofweek >= 5 else 0
                promo = 1 if np.random.rand() < 0.05 else 0
                promo_effect = promo * np.random.uniform(30, 60)
                noise = np.random.normal(0, 8)

                demand = (store_base * cat_multiplier) + trend + yearly_season + weekly_season + promo_effect + noise
                demand = max(0, round(demand))

                rows.append({
                    "date": date, "store": store, "category": category,
                    "day_of_week": date.dayofweek, "month": date.month,
                    "is_weekend": int(date.dayofweek >= 5), "promo": promo,
                    "units_sold": demand
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    if REAL_DATA_PATH.exists():
        df = load_real_data(REAL_DATA_PATH)
    else:
        df = generate_synthetic_data()

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows -> {OUTPUT_PATH}")
    print(f"Stores: {df['store'].nunique()} | Categories/Items: {df['category'].nunique()} "
          f"| Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.head())