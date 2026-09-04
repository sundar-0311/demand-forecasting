"""
Demand Forecasting Pipeline
---------------------------
1. Load and clean daily sales data (multi-store, multi-category/item)
2. Feature engineering: calendar features, lag features, rolling statistics
3. Time-based train/test split (no shuffling — respects temporal order)
4. Train and compare: Naive baseline, Linear Regression, Random Forest, XGBoost
5. Hyperparameter-tune XGBoost using RandomizedSearchCV + TimeSeriesSplit
   (proper time-series CV — never trains on future data to predict the past)
6. Evaluate all models with MAE, RMSE, MAPE; report tuned vs untuned XGBoost
7. Save the best model + feature importance plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("data/sales_data.csv", parse_dates=["date"])
df = df.sort_values(["store", "category", "date"]).reset_index(drop=True)

# ---------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------
group_cols = ["store", "category"]

grouped_target = df.groupby(group_cols)["units_sold"]
for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = grouped_target.shift(lag)

shifted = grouped_target.shift(1)
for window in [7, 14, 28]:
    df[f"rolling_mean_{window}"] = shifted.groupby([df["store"], df["category"]]).rolling(window).mean().reset_index(level=group_cols, drop=True)
    df[f"rolling_std_{window}"] = shifted.groupby([df["store"], df["category"]]).rolling(window).std().reset_index(level=group_cols, drop=True)

df["day_of_month"] = df["date"].dt.day
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["quarter"] = df["date"].dt.quarter

df = df.dropna().reset_index(drop=True)

feature_cols = [
    "day_of_week", "month", "is_weekend", "promo", "day_of_month",
    "week_of_year", "quarter",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_14", "rolling_std_28",
]
categorical_cols = ["store", "category"]
target_col = "units_sold"

X = df[feature_cols + categorical_cols]
y = df[target_col]

# ---------------------------------------------------------------
# 3. Time-based train/test split (last 60 days per series = test set)
# ---------------------------------------------------------------
cutoff_date = df["date"].max() - pd.Timedelta(days=60)
train_mask = df["date"] <= cutoff_date
test_mask = df["date"] > cutoff_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
dates_test = df.loc[test_mask, "date"]

print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
print(f"Train period: up to {cutoff_date.date()} | Test period: {cutoff_date.date()} to {df['date'].max().date()}")

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
    }

results = {}

# ---------------------------------------------------------------
# 4. Baseline + untuned models
# ---------------------------------------------------------------
results["Naive (lag-7)"] = evaluate(y_test, X_test["lag_7"])

lr_pipeline = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
lr_pipeline.fit(X_train, y_train)
results["Linear Regression"] = evaluate(y_test, lr_pipeline.predict(X_test))

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1))
])
rf_pipeline.fit(X_train, y_train)
results["Random Forest (default)"] = evaluate(y_test, rf_pipeline.predict(X_test))

xgb_default_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    ))
])
xgb_default_pipeline.fit(X_train, y_train)
results["XGBoost (default)"] = evaluate(y_test, xgb_default_pipeline.predict(X_test))

# ---------------------------------------------------------------
# 5. Hyperparameter tuning: XGBoost with TimeSeriesSplit CV
#    TimeSeriesSplit ensures every validation fold is strictly AFTER
#    its training fold in time — a regular KFold shuffle would let the
#    model "see the future" during cross-validation, which silently
#    inflates CV scores and doesn't reflect real forecasting conditions.
# ---------------------------------------------------------------
print("\nRunning hyperparameter search for XGBoost (this may take a minute)...")

param_distributions = {
    "model__n_estimators": [200, 300, 400, 600, 800],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5, 7],
}

xgb_search_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(random_state=42))
])

tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=xgb_search_pipeline,
    param_distributions=param_distributions,
    n_iter=25,
    scoring="neg_root_mean_squared_error",
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)

best_xgb_pipeline = search.best_estimator_
tuned_pred = best_xgb_pipeline.predict(X_test)
results["XGBoost (tuned)"] = evaluate(y_test, tuned_pred)

print(f"\nBest CV RMSE (time-series CV, train set only): {-search.best_score_:.2f}")
print("Best hyperparameters found:")
for param, value in search.best_params_.items():
    print(f"  {param.replace('model__', '')}: {value}")

# ---------------------------------------------------------------
# 6. Results comparison
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).T.sort_values("RMSE")
print("\n=== Model Comparison (Test Set, last 60 days) ===")
print(results_df.round(2))
results_df.to_csv("reports/model_comparison.csv")

improvement_pct = (
    (results["XGBoost (default)"]["RMSE"] - results["XGBoost (tuned)"]["RMSE"])
    / results["XGBoost (default)"]["RMSE"] * 100
)
print(f"\nTuning improved XGBoost RMSE by {improvement_pct:.1f}% versus default hyperparameters.")

best_model_name = results_df["RMSE"].idxmin()
print(f"Best overall model: {best_model_name}")

# ---------------------------------------------------------------
# 7. Save best model + feature importance plot
# ---------------------------------------------------------------
joblib.dump(best_xgb_pipeline, "models/xgb_demand_model.pkl")
joblib.dump(rf_pipeline, "models/rf_demand_model.pkl")

ohe_feature_names = best_xgb_pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_cols)
all_feature_names = list(ohe_feature_names) + feature_cols
importances = best_xgb_pipeline.named_steps["model"].feature_importances_

imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances}).sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(8, 6))
plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#2E86AB")
plt.title("Top 15 Feature Importances (Tuned XGBoost)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=150)
print("Saved: reports/feature_importance.png")

sample_store = df.loc[test_mask, "store"].iloc[0]
sample_category = df.loc[test_mask, "category"].iloc[0]
sample_mask = (df.loc[test_mask, "store"] == sample_store) & (df.loc[test_mask, "category"] == sample_category)

plt.figure(figsize=(10, 5))
plt.plot(dates_test[sample_mask], y_test[sample_mask].values, label="Actual", marker="o", markersize=3)
plt.plot(dates_test[sample_mask], tuned_pred[sample_mask.values], label="Tuned XGBoost Predicted", marker="x", markersize=3)
plt.title(f"Actual vs Predicted Demand — {sample_store} / {sample_category} (Test Period)")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/actual_vs_predicted.png", dpi=150)
print("Saved: reports/actual_vs_predicted.png")