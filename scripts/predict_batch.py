"""Run batch predictions on ClickPrediction and write CSV output."""

from __future__ import annotations

import argparse
from pathlib import Path

from deliveroo_click_prediction.config import PROCESSED_DATA_DIR, TARGET_COL
from deliveroo_click_prediction.data_loader import load_local_rdata
from deliveroo_click_prediction.model import train_xgb_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch prediction runner.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "batch_predictions.csv",
        help="Output CSV file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = load_local_rdata()
    df_train = datasets["ClickTraining"].copy()
    df_pred = datasets["ClickPrediction"].copy()

    x_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL].astype(int)
    model = train_xgb_model(x_train, y_train, val_size=0.2)["model"]

    probs = model.predict_proba(df_pred)[:, 1]
    output_df = df_pred.copy()
    output_df["Click_Probability"] = probs
    output_df["Predicted_Click"] = (probs > 0.5).astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
