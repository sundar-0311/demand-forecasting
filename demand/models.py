"""Model factory and hyper-parameter search spaces."""

from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .config import SEED

ML_MODELS = ["Linear Regression", "Random Forest", "XGBoost", "LightGBM"]

SEARCH_SPACES = {
    "LightGBM": {
        "n_estimators": [300, 500, 800, 1200],
        "learning_rate": [0.02, 0.05, 0.1],
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 10.0],
    },
    "XGBoost": {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    },
}


def build_model(name: str, params: dict | None = None):
    params = params or {}
    if name == "LightGBM":
        return LGBMRegressor(**{"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
                                "subsample_freq": 1, **params},
                             random_state=SEED, n_jobs=-1, verbose=-1)
    if name == "XGBoost":
        return XGBRegressor(**{"n_estimators": 500, "learning_rate": 0.05, "max_depth": 8, **params},
                            tree_method="hist", random_state=SEED, n_jobs=-1)
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=80, max_depth=16, min_samples_leaf=5,
                                     max_features=0.6, max_samples=0.5,
                                     random_state=SEED, n_jobs=-1)
    if name == "Linear Regression":
        encode_ids = ColumnTransformer(
            [("ids", OneHotEncoder(handle_unknown="ignore"), ["store", "item"])],
            remainder="passthrough",
        )
        return Pipeline([("encode", encode_ids), ("model", LinearRegression())])
    raise ValueError(f"Unknown model {name!r}")
