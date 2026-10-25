
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path("data") / "train_with_rul.csv")

# Plot the RUL of the first engine
engine = df[df["engine_id"] == 1]

plt.figure(figsize=(10, 5))
plt.plot(engine["cycle"], engine["RUL"])
plt.xlabel("Operating Cycle")
plt.ylabel("Remaining Useful Life")
plt.title("Engine 1 - Remaining Useful Life")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot one sensor over operating cycles
plt.figure(figsize=(10, 5))
plt.plot(engine["cycle"], engine["sensor_2"])
plt.xlabel("Operating Cycle")
plt.ylabel("Sensor 2 Reading")
plt.title("Engine 1 - Sensor 2 Trend")
plt.grid(True)
plt.tight_layout()
plt.show()
