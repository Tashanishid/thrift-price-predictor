"""
train.py — Model Training Pipeline
------------------------------------
Loads the merged, cleaned dataset produced by data_prep.py, trains a
Random Forest Regressor, evaluates it, and saves the model to disk.
"""

import os
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH  = "data/processed/fashion_training_data.csv"
MODEL_PATH = "models/price_predictor.pkl"

# Explicit feature list — must match the columns written by data_prep.py.
# Using an explicit list (rather than df.drop("price")) ensures no stray
# columns from any source dataset accidentally become model features.
FEATURE_COLS = [
    "brand", "category", "condition", "size",   # base features
    "brand_tier", "is_vintage", "season",        # engineered categoricals
    "retail_price_ratio",                        # engineered numeric
]
TARGET_COL   = "price"


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Processed data not found at '{DATA_PATH}'.\n"
            "Run: python src/data_prep.py"
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows from '{DATA_PATH}'")
    return df


def train_model():
    df = load_data()

    X = df[FEATURE_COLS]

    # Log-transform the target so the model learns proportional errors rather
    # than absolute ones. A $5 miss on a $10 item matters more than a $5 miss
    # on a $200 item — log scale treats them proportionally.
    # np.log1p(x) = log(1 + x), safe when x == 0 (avoids log(0) = -inf).
    y = np.log1p(df[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    print("Training…")
    model.fit(X_train, y_train)

    # Convert log-scale predictions back to dollar prices for interpretable metrics.
    # np.expm1(x) = exp(x) - 1, the exact inverse of log1p.
    log_preds   = model.predict(X_test)
    preds_price = np.expm1(log_preds)
    y_test_price = np.expm1(y_test)

    print("\n── Evaluation (dollar scale) ──")
    print(f"MAE  : ${mean_absolute_error(y_test_price, preds_price):.2f}")
    print(f"RMSE : ${mean_squared_error(y_test_price, preds_price) ** 0.5:.2f}")
    print(f"R²   : {r2_score(y_test, log_preds):.4f}  (log scale — model's actual target)")

    print("\n── Feature importances ──")
    for col, imp in sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    ):
        print(f"  {col:<12}: {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved → '{MODEL_PATH}'")


if __name__ == "__main__":
    train_model()
