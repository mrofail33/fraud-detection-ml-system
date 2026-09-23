from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_detection.config import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, TARGET_COLUMN
from fraud_detection.data import clean_target, load_dataset, make_train_test_split, select_features
from fraud_detection.modeling import train_and_evaluate_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Save raw confusion matrices for the fraud benchmark.")
    parser.add_argument("--download-openml", action="store_true")
    parser.add_argument("--data-path")
    parser.add_argument("--output", default="outputs/real-run-2026-09-23/confusion_matrices.json")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    df = load_dataset(data_path=args.data_path, download_openml=args.download_openml)
    df = clean_target(df)
    feature_columns = select_features(df)
    x_train, x_test, y_train, y_test = make_train_test_split(
        df,
        feature_columns=feature_columns,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    trained_models, metrics = train_and_evaluate_models(
        x_train,
        x_test,
        y_train,
        y_test,
        random_state=args.random_state,
    )

    matrices = {}
    for model_name, model in trained_models.items():
        tn, fp, fn, tp = confusion_matrix(y_test, model.predict(x_test), labels=[0, 1]).ravel()
        matrices[model_name] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    summary = {
        "dataset_rows": int(len(df)),
        "fraud_count": int(df[TARGET_COLUMN].sum()),
        "fraud_rate": float(df[TARGET_COLUMN].mean()),
        "test_rows": int(len(y_test)),
        "test_fraud_count": int(y_test.sum()),
        "best_model_by_f1": metrics.iloc[0]["model"],
        "confusion_matrices": matrices,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
