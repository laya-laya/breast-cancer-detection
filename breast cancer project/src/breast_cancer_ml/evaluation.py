"""Cross-validation and held-out evaluation utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
}


def cross_validate_models(
    models: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare candidate models using stratified cross-validation."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    rows: list[dict[str, float | str]] = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
        )
        row: dict[str, float | str] = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_test(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Evaluate a fitted binary classifier on a held-out set."""
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "sensitivity_recall": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }
