# Deliveroo Click Prediction

## Description
This project develops a machine learning model to predict whether users will click on Deliveroo advertisements using session-level behavioral and contextual data. The goal is to identify high-intent users and support data-driven marketing decisions such as ad targeting and budget allocation. Multiple classification models were evaluated, with XGBoost achieving the best predictive performance (AUC ≈ 0.985). The results demonstrate strong capability to distinguish potential converters and provide actionable insights to optimize campaign effectiveness.

## Key Dependencies
- XGBoost: Gradient boosting model used as the main predictor for click probability.
- Scikit-learn: Utilities for training workflows, evaluation metrics, and ML tooling.
- Streamlit: Framework used to serve the interactive prediction app.
- Plotly: Interactive visualizations for model metrics and business insights in the UI.
- Pyreadr: Bridge for loading `.RData` inputs into the Python pipeline.
- Joblib: Serialization of trained model artifacts for reuse in inference and app flows.

## Setup
```bash
poetry install
poetry shell
```

## Usage
Run the Streamlit app:
```bash
streamlit run scripts/run_app.py
```

## Structure
- `src/deliveroo_click_prediction/`: importable package code (config, data loading, preprocessing, model, app modules).
- `scripts/`: runnable entrypoints for training, prediction, app launch, and R analysis.
- `data/raw/`: raw input datasets.
- `data/processed/`: generated outputs (for example, prediction CSV files).
- `docs/`: project presentation.

## Notes
- Put `DeliveryAdClick.RData` in `data/raw/`.
- Python workflows expect R objects named `ClickTraining` and `ClickPrediction`.
- The project targets Python `>=3.12,<3.13`.
