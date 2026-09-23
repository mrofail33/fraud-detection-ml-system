from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_detection.config import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    MODEL_FILENAME,
    MODELS_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
    TARGET_COLUMN,
)
from fraud_detection.data import (
    clean_target,
    load_dataset,
    make_train_test_split,
    select_features,
)
from fraud_detection.eda import run_eda
from fraud_detection.modeling import (
    cross_validate_models,
    evaluate_all_legitimate_baseline,
    get_feature_importance,
    plot_confusion_matrices,
    plot_roc_curves,
    save_model,
    train_and_evaluate_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection ML models.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to creditcard.csv.")
    parser.add_argument(
        "--download-openml",
        action="store_true",
        help="Download the public Credit Card Fraud dataset from OpenML dataset ID 1597.",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use a small generated dataset for a quick smoke test.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for faster runs.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(
        data_path=args.data_path,
        download_openml=args.download_openml,
        use_sample=args.use_sample,
        random_state=args.random_state,
    )
    df = clean_target(df)

    if args.max_rows and len(df) > args.max_rows:
        sample_fraction = args.max_rows / len(df)
        df = (
            df.groupby(TARGET_COLUMN, group_keys=False)
            .sample(frac=sample_fraction, random_state=args.random_state)
            .sample(frac=1, random_state=args.random_state)
            .reset_index(drop=True)
        )

    feature_columns = select_features(df)
    run_eda(df, feature_columns, plots_dir)

    X_train, X_test, y_train, y_test = make_train_test_split(
        df,
        feature_columns=feature_columns,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    cv_metrics = cross_validate_models(
        df[feature_columns],
        df[TARGET_COLUMN],
        random_state=args.random_state,
    )

    trained_models, metrics = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, random_state=args.random_state
    )
    baseline = evaluate_all_legitimate_baseline(y_test)
    comparison = pd.concat([pd.DataFrame([baseline]), metrics], ignore_index=True)

    metrics_path = output_dir / "model_metrics.csv"
    comparison_path = output_dir / "model_metrics_with_baseline.csv"
    cv_path = output_dir / "cross_validation_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    cv_metrics.to_csv(cv_path, index=False)

    best_model_name = metrics.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    save_model(best_model, feature_columns, models_dir / MODEL_FILENAME)
    feature_importance = get_feature_importance(best_model, feature_columns)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    plot_confusion_matrices(trained_models, X_test, y_test, plots_dir)
    plot_roc_curves(trained_models, X_test, y_test, plots_dir / "roc_curves.png")

    summary = {
        "rows": int(len(df)),
        "features": feature_columns,
        "target_column": TARGET_COLUMN,
        "fraud_rate": float(df[TARGET_COLUMN].mean()),
        "best_model": best_model_name,
        "metrics_path": str(metrics_path),
        "comparison_path": str(comparison_path),
        "cross_validation_path": str(cv_path),
        "feature_importance_path": str(output_dir / "feature_importance.csv"),
        "model_path": str(models_dir / MODEL_FILENAME),
        "plots_dir": str(plots_dir),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Training complete.")
    print(f"Best model by F1 score: {best_model_name}")
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
