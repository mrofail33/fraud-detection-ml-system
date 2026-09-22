from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import DEFAULT_RANDOM_STATE


def build_models(random_state: int = DEFAULT_RANDOM_STATE) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=10,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.08,
                        max_depth=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test, random_state: int):
    results = []
    trained_models = {}

    for model_name, pipeline in build_models(random_state).items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = get_positive_class_scores(pipeline, X_test)

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_score),
            }
        )
        trained_models[model_name] = pipeline

    metrics = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
    return trained_models, metrics


def evaluate_all_legitimate_baseline(y_test) -> dict[str, float]:
    y_pred = [0] * len(y_test)
    y_score = [0] * len(y_test)
    return {
        "model": "Always Legitimate Baseline",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score) if len(set(y_test)) > 1 else 0.5,
    }


def get_positive_class_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def save_model(model, feature_columns: list[str], output_path: str | Path) -> None:
    payload = {
        "model": model,
        "feature_columns": feature_columns,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)


def load_model(model_path: str | Path):
    return joblib.load(model_path)


def plot_confusion_matrices(models: dict, X_test, y_test, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred)
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Legitimate", "Fraud"],
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        display.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{model_name} Confusion Matrix")
        fig.tight_layout()
        file_name = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
        fig.savefig(output_path / file_name, dpi=150)
        plt.close(fig)


def plot_roc_curves(models: dict, X_test, y_test, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, model in models.items():
        y_score = get_positive_class_scores(model, X_test)
        RocCurveDisplay.from_predictions(y_test, y_score, name=model_name, ax=ax)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
