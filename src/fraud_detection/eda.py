from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_detection.config import TARGET_COLUMN


def run_eda(df: pd.DataFrame, feature_columns: list[str], output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_paths = {
        "class_balance": output_path / "class_balance.png",
        "amount_distribution": output_path / "amount_distribution.png",
        "correlation_heatmap": output_path / "correlation_heatmap.png",
        "top_feature_correlations": output_path / "top_feature_correlations.png",
    }

    plot_class_balance(df, plot_paths["class_balance"])
    plot_amount_distribution(df, plot_paths["amount_distribution"])
    plot_correlation_heatmap(df, feature_columns, plot_paths["correlation_heatmap"])
    plot_top_feature_correlations(df, feature_columns, plot_paths["top_feature_correlations"])

    return {name: str(path) for name, path in plot_paths.items()}


def plot_class_balance(df: pd.DataFrame, output_path: Path) -> None:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#4C78A8", "#E45756"])
    ax.set_title("Fraud vs Legitimate Transactions")
    ax.set_ylabel("Number of transactions")
    for index, value in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_amount_distribution(df: pd.DataFrame, output_path: Path) -> None:
    if "Amount" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label_value, label, color in [(0, "Legitimate", "#4C78A8"), (1, "Fraud", "#E45756")]:
        subset = df.loc[df[TARGET_COLUMN] == label_value, "Amount"].dropna()
        if not subset.empty:
            ax.hist(np.log1p(subset), bins=40, alpha=0.65, label=label, color=color)

    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("log(Amount + 1)")
    ax.set_ylabel("Number of transactions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, feature_columns: list[str], output_path: Path) -> None:
    selected = choose_correlation_features(df, feature_columns, max_features=12)
    corr = df[selected + [TARGET_COLUMN]].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_top_feature_correlations(df: pd.DataFrame, feature_columns: list[str], output_path: Path) -> None:
    correlations = (
        df[feature_columns + [TARGET_COLUMN]]
        .corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .dropna()
        .abs()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(correlations.index, correlations.values, color="#72B7B2")
    ax.set_title("Most Useful Features by Correlation With Fraud")
    ax.set_xlabel("Absolute correlation with fraud label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def choose_correlation_features(
    df: pd.DataFrame, feature_columns: list[str], max_features: int = 12
) -> list[str]:
    correlations = (
        df[feature_columns + [TARGET_COLUMN]]
        .corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .dropna()
        .abs()
        .sort_values(ascending=False)
    )
    return correlations.head(max_features).index.tolist()
