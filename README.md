# Deliveroo Click Prediction

## Description
This project develops a machine learning model to predict whether users will click on Deliveroo advertisements using session-level behavioral and contextual data. The goal is to identify high-intent users and support data-driven marketing decisions such as ad targeting and budget allocation. Multiple classification models were evaluated, with XGBoost achieving the best predictive performance (AUC ≈ 0.985). The results demonstrate strong capability to distinguish potential converters and provide actionable insights to optimize campaign effectiveness.

## Setup
```bash
poetry install
poetry shell
```

## Usage
Train the model:
```bash
python scripts/train.py
```

Run batch prediction (default output: `data/processed/batch_predictions.csv`):
```bash
python scripts/predict_batch.py
```

Run the Streamlit app:
```bash
streamlit run scripts/run_app.py
```

Minimal import example:
```python
from deliveroo_click_prediction.model import train_xgb_model
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
