
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupShuffleSplit

# Load data and model
df = pd.read_csv("data/train_with_rul.csv")
saved = joblib.load("models/random_forest_rul.pkl")

model = saved["model"]
feature_cols = saved["feature_cols"]

X = df[feature_cols]
y = df["RUL"]

# Reproduce the same engine-level validation split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

_, val_idx = next(
    splitter.split(X, y, groups=df["engine_id"])
)

X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]
predictions = model.predict(X_val)

# Actual versus predicted RUL
plt.figure(figsize=(7, 6))
plt.scatter(y_val, predictions, alpha=0.25)
plt.plot(
    [0, y_val.max()],
    [0, y_val.max()],
    linestyle="--"
)
plt.xlabel("Actual RUL (cycles)")
plt.ylabel("Predicted RUL (cycles)")
plt.title("Actual vs Predicted Remaining Useful Life")
plt.tight_layout()
plt.show()

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False).head(10)

importance.sort_values().plot(
    kind="barh",
    figsize=(8, 5)
)
plt.xlabel("Importance")
plt.title("Top 10 Features for RUL Prediction")
plt.tight_layout()
plt.show()
