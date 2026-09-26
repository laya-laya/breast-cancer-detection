"""Run the complete reproducible training and evaluation workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from .data import make_split
from .evaluation import cross_validate_models, evaluate_test
from .models import get_models


def project_root() -> Path:
    """Return the repository root for an editable/source checkout."""
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = project_root()
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = make_split()
    models = get_models()

    cv_results = cross_validate_models(models, X_train, y_train)
    cv_results.to_csv(results_dir / "cross_validation.csv", index=False)

    fitted: dict[str, object] = {}
    test_rows: list[dict[str, float | int | str]] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        test_rows.append({"model": name, **evaluate_test(model, X_test, y_test)})

    test_results = (
        pd.DataFrame(test_rows)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )
    test_results.to_csv(results_dir / "test_metrics.csv", index=False)

    # Model choice is based on training-set CV only, never on held-out test metrics.
    best_name = str(cv_results.iloc[0]["model"])
    best_model = fitted[best_name]

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("ROC curves — held-out test set")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in fitted.items():
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("Precision–recall curves — held-out test set")
    fig.tight_layout()
    fig.savefig(figures_dir / "precision_recall_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        labels=[0, 1],
        display_labels=["benign", "malignant"],
        ax=ax,
    )
    ax.set_title(f"Confusion matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Post-hoc interpretation on the held-out set. Because several WDBC features
    # are strongly correlated, permutation importance should not be interpreted
    # as a unique causal ranking of biological features.
    importance_result = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": importance_result.importances_mean,
                "importance_std": importance_result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(results_dir / "permutation_importance.csv", index=False)

    top = importance.head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
    )
    ax.set_xlabel("Decrease in ROC-AUC after permutation")
    ax.set_title("Permutation importance — selected model")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=180)
    plt.close(fig)

    print("\nCross-validation results\n")
    print(cv_results.to_string(index=False))
    print("\nHeld-out test results\n")
    print(test_results.to_string(index=False))
    print(f"\nSelected by training CV ROC-AUC: {best_name}")


if __name__ == "__main__":
    main()
