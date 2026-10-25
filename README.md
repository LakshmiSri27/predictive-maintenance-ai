
# Predictive Maintenance AI: Remaining Useful Life Estimation

## Overview

Predictive Maintenance AI is a machine learning project that estimates the Remaining Useful Life (RUL) of an engine using sensor measurements.

The project uses the NASA C-MAPSS Turbofan Engine Degradation dataset to explore engine degradation patterns, train regression models, and support data-driven maintenance planning.

## Problem Statement

Unexpected equipment failures can cause operational delays and expensive repairs. This project investigates whether historical sensor measurements can be used to estimate the remaining operating cycles of an engine.

## Objectives

- Analyze engine sensor and operating-cycle data.
- Calculate Remaining Useful Life (RUL) targets.
- Train and compare Random Forest and XGBoost regression models.
- Evaluate predictions using MAE and RMSE.
- Visualize engine degradation trends.
- Deploy an interactive Streamlit application.

## Dataset

Dataset: NASA C-MAPSS Turbofan Engine Degradation Dataset.

Download:
https://zenodo.org/records/15346912

The FD001 training, test, and RUL files should be placed inside the `data/` directory.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Regressor
- XGBoost Regressor
- Matplotlib
- Streamlit
- Joblib

## Project Structure

```text
predictive-maintenance-ai/
├── data/
├── models/
├── app.py
├── train.py
├── explore.py
├── model_training.py
├── compare_models.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Methodology

1. Load and inspect the engine sensor dataset.
2. Calculate RUL from the final recorded cycle of each training engine.
3. Explore RUL and sensor trends.
4. Split training and validation records by engine.
5. Train Random Forest and XGBoost regression models.
6. Compare performance using MAE and RMSE.
7. Save the best validation model.
8. Visualize results and run predictions through Streamlit.

## Model Evaluation

Validation results from the current experiment:

| Model | MAE (cycles) | RMSE (cycles) |
|---|---:|---:|
| Random Forest | 25.804 | 35.286 |
| XGBoost | 25.951 | 35.678 |

Random Forest achieved the lower MAE and RMSE in this experiment.

These values are from the current validation split, not from an independent final test evaluation.

## Run Locally

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train and compare models

```bash
python train.py
python compare_models.py
```

Run the other scripts as needed:

```bash
python explore.py
python evaluate.py
```

### 4. Launch the application

```bash
python -m streamlit run app.py
```

## Limitations

- The dataset contains simulated engine degradation data.
- Predictions depend on the quality and representativeness of sensor readings.
- The current model is a research demonstration and is not validated for real aircraft maintenance or safety decisions.
- Final performance should be evaluated using the supplied test data and its RUL labels.

## Future Enhancements

- Add LSTM-based sequence modelling.
- Improve sensor feature engineering.
- Add model explainability and uncertainty estimates.
- Evaluate predictions on the official test set.
- Improve input validation and interactive visualizations.

## Author

Lakshmi Sri V

## License and Dataset Attribution

Consult the original dataset page for its applicable terms and attribution requirements before redistributing the data.
