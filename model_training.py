
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

# Paths
DATA_PATH = Path("data/train_with_rul.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load processed data
df = pd.read_csv(DATA_PATH)

# Select sensor and operating setting features
feature_cols = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

X = df[feature_cols]
y = df["RUL"]
groups = df["engine_id"]

# Split by engine to reduce data leakage
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

# Train model
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("\nRandom Forest Results")
print("---------------------")
print(f"MAE:  {mae:.2f} cycles")
print(f"RMSE: {rmse:.2f} cycles")

# Save model and feature list
joblib.dump(
    {"model": model, "feature_cols": feature_cols},
    MODEL_DIR / "random_forest_rul.pkl"
)

print("\nModel saved to models/random_forest_rul.pkl")
