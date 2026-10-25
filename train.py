
import pandas as pd
import numpy as np
from pathlib import Path

# Dataset location
DATA_DIR = Path("data")

# Dataset column names
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load training data
train_path = DATA_DIR / "train_FD001.txt"

df = pd.read_csv(
    train_path,
    sep=r"\s+",
    header=None,
    names=columns
)

# Calculate the last recorded cycle for each engine
last_cycle = df.groupby("engine_id")["cycle"].max()

# Calculate Remaining Useful Life (RUL)
df["RUL"] = (
    df["engine_id"].map(last_cycle) - df["cycle"]
)

print("RUL calculation completed!")
print("\nDataset shape:", df.shape)
print("\nSample RUL values:")
print(df[["engine_id", "cycle", "RUL"]].head(10))

print("\nRUL statistics:")
print(df["RUL"].describe())

# Save processed dataset
output_path = DATA_DIR / "train_with_rul.csv"
df.to_csv(output_path, index=False)

print(f"\nProcessed dataset saved to: {output_path}")
