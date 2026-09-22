from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.config import MODEL_FILENAME, MODELS_DIR
from fraud_detection.modeling import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fraud probability for one transaction.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODELS_DIR / MODEL_FILENAME),
        help="Path to saved joblib model payload.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help='Transaction as JSON, for example: {"Time": 1000, "Amount": 3800, "V1": 0.2}',
    )
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--amount", type=float, default=None)
    return parser.parse_args()


def make_input_row(args: argparse.Namespace, feature_columns: list[str]) -> pd.DataFrame:
    row = {}
    if args.json:
        row.update(json.loads(args.json))
    if args.time is not None:
        row["Time"] = args.time
    if args.amount is not None:
        row["Amount"] = args.amount

    aligned = {column: row.get(column, np.nan) for column in feature_columns}
    return pd.DataFrame([aligned], columns=feature_columns)


def main() -> None:
    args = parse_args()
    payload = load_model(Path(args.model_path))
    model = payload["model"]
    feature_columns = payload["feature_columns"]

    X = make_input_row(args, feature_columns)
    probability = model.predict_proba(X)[0, 1]
    label = int(probability >= 0.5)

    print(f"Fraud probability: {probability:.2%}")
    print(f"Prediction: {'Potential fraud' if label else 'Likely legitimate'}")


if __name__ == "__main__":
    main()
