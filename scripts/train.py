"""Train the click prediction model from local raw data."""

from __future__ import annotations

import argparse

import pandas as pd

from deliveroo_click_prediction.config import TARGET_COL
from deliveroo_click_prediction.data_loader import load_local_rdata
from deliveroo_click_prediction.model import train_xgb_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Deliveroo click prediction model.")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation split size (default: 0.2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = load_local_rdata()
    df_train = datasets["ClickTraining"].copy()

    x_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL].astype(int)

    artifacts = train_xgb_model(x_train, y_train, val_size=args.val_size)

    metrics = artifacts["metrics"]
    print("Training finished")
    for key in ["AUC", "Accuracy", "F1_score", "Precision", "Recall"]:
        print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
