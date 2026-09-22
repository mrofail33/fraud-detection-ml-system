from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from fraud_detection.config import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, TARGET_COLUMN


def load_dataset(
    data_path: str | Path | None = None,
    download_openml: bool = False,
    use_sample: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Load the fraud dataset from CSV, OpenML, or a small generated demo sample."""
    if data_path:
        return pd.read_csv(data_path)

    if download_openml:
        dataset = fetch_openml(data_id=1597, as_frame=True, parser="auto")
        frame = dataset.frame.copy()
        if "class" in frame.columns and TARGET_COLUMN not in frame.columns:
            frame = frame.rename(columns={"class": TARGET_COLUMN})
        return frame

    if use_sample:
        return make_demo_dataset(random_state=random_state)

    raise ValueError(
        "Provide --data-path, use --download-openml, or use --use-sample for a quick demo run."
    )


def make_demo_dataset(n_rows: int = 3000, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Create a tiny imbalanced fraud-like dataset for tests and demos.

    This is not a replacement for the public credit card fraud dataset. It only
    lets the full workflow run quickly on a fresh machine.
    """
    rng = np.random.default_rng(random_state)
    fraud_rate = 0.025
    y = rng.binomial(1, fraud_rate, size=n_rows)

    amount = rng.gamma(shape=2.0, scale=45.0, size=n_rows)
    amount[y == 1] += rng.gamma(shape=4.0, scale=90.0, size=(y == 1).sum())

    time = rng.integers(0, 172800, size=n_rows)
    data = {
        "Time": time,
        "Amount": amount,
        TARGET_COLUMN: y,
    }

    for i in range(1, 29):
        signal = rng.normal(0, 1, size=n_rows)
        if i in {3, 7, 10, 14, 17}:
            signal += y * rng.normal(1.3, 0.4, size=n_rows)
        data[f"V{i}"] = signal

    frame = pd.DataFrame(data)
    missing_indices = rng.choice(frame.index, size=max(1, n_rows // 100), replace=False)
    frame.loc[missing_indices, "Amount"] = np.nan
    return frame


def clean_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Drop rows without a target and ensure the target is integer encoded."""
    if target_column not in df.columns:
        raise ValueError(f"Expected target column '{target_column}' in dataset.")

    cleaned = df.dropna(subset=[target_column]).copy()
    cleaned[target_column] = cleaned[target_column].astype(int)
    return cleaned


def select_features(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[str]:
    """Select numeric model features while excluding the target."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [column for column in numeric_columns if column != target_column]

    if not features:
        raise ValueError("No numeric feature columns were found.")

    return features


def make_train_test_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    X = df[feature_columns]
    y = df[target_column]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
