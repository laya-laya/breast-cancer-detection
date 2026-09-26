"""Dataset loading and deterministic train/test splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load WDBC and encode malignant tumors as the positive class (1).

    Scikit-learn's original target encoding is 0=malignant and 1=benign.
    Re-encoding malignant as 1 makes sensitivity/recall and false-negative
    counts easier to interpret.
    """
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = (dataset.target == 0).astype(int)
    y.name = "malignant"
    return X, y


def make_split(
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split."""
    X, y = load_data()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
