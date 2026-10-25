
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Paths
DATA_PATH = Path("data/train_with_rul.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

feature_cols = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

X = df[feature_cols]
y = df["RUL"]
groups = df["engine_id"]

# Split by engine to avoid engine overlap
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

# Define models
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3)
    })

    trained_models[name] = model
    print(f"MAE: {mae:.3f} cycles")
    print(f"RMSE: {rmse:.3f} cycles")

# Compare models
results_df = pd.DataFrame(results).sort_values("MAE")
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

# Select the model with the lower validation MAE
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump(
    {
        "model": best_model,
        "model_name": best_name,
        "feature_cols": feature_cols
    },
    MODEL_DIR / "best_rul_model.pkl"
)

print(f"\nBest model by validation MAE: {best_name}")
print("Saved to models/best_rul_model.pkl")
