
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

MODEL_PATH = Path("models/best_rul_model.pkl")
DATA_PATH = Path("data/train_with_rul.csv")

st.title("Predictive Maintenance AI")
st.write(
    "Estimate the Remaining Useful Life (RUL) "
    "of an engine using sensor measurements."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Train the model first.")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved["model"]
feature_cols = saved["feature_cols"]

st.sidebar.header("Engine Sensor Inputs")
st.sidebar.caption(
    "Enter sensor readings from a single operating cycle."
)

values = {}

for col in feature_cols:
    if col.startswith("setting_"):
        default = 0.0
    else:
        default = 0.0

    values[col] = st.sidebar.number_input(
        col,
        value=default,
        format="%.4f"
    )

if st.button("Predict Remaining Useful Life", type="primary"):
    input_df = pd.DataFrame([values])[feature_cols]
    prediction = max(0.0, float(model.predict(input_df)[0]))

    st.subheader("Prediction Result")
    st.metric(
        "Estimated Remaining Useful Life",
        f"{prediction:.1f} cycles"
    )

    if prediction <= 20:
        st.warning(
            "The model predicts relatively low remaining life. "
            "Further inspection may be appropriate."
        )
    else:
        st.info(
            "Review this estimate alongside sensor history "
            "and maintenance records."
        )

st.divider()
st.subheader("Training Data Overview")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

    c1, c2, c3 = st.columns(3)
    c1.metric("Training Records", f"{len(df):,}")
    c2.metric("Engines", df["engine_id"].nunique())
    c3.metric("Sensor Features", len(feature_cols))

    selected_engine = st.selectbox(
        "Choose an engine to explore",
        sorted(df["engine_id"].unique())
    )

    engine_df = df[df["engine_id"] == selected_engine]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(engine_df["cycle"], engine_df["RUL"])
    ax.set_xlabel("Operating Cycle")
    ax.set_ylabel("Remaining Useful Life")
    ax.set_title(f"Engine {selected_engine}: RUL Trend")
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Training data CSV is not available.")
