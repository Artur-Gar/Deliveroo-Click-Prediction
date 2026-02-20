"""Model definition and training logic."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from deliveroo_click_prediction.preprocessing import build_preprocessor


def build_xgb_pipeline(x: pd.DataFrame) -> Pipeline:
    """Build preprocessing + classifier pipeline."""
    preprocessor = build_preprocessor(x)
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", model),
        ]
    )


def train_xgb_model(x: pd.DataFrame, y: pd.Series, val_size: float = 0.2) -> dict[str, Any]:
    """Train the XGBoost pipeline and return model artifacts."""
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=val_size,
        random_state=42,
        stratify=y,
    )

    pipeline = build_xgb_pipeline(x_train)
    param_grid = {
        "classifier__learning_rate": [0.1],
        "classifier__n_estimators": [100],
        "classifier__max_depth": [3],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(x_train, y_train)

    model = grid.best_estimator_
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1_score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_prob),
    }

    return {
        "model": model,
        "metrics": metrics,
        "X_test": x_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
