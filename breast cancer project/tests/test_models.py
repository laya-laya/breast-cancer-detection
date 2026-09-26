import numpy as np

from breast_cancer_ml.data import make_split
from breast_cancer_ml.evaluation import cross_validate_models, evaluate_test
from breast_cancer_ml.models import get_models


def test_models_fit_and_predict_probabilities():
    X_train, X_test, y_train, _ = make_split()
    for model in get_models().values():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test.iloc[:3])
        assert probability.shape == (3, 2)
        assert np.all((probability >= 0) & (probability <= 1))


def test_evaluation_metrics_are_valid():
    X_train, X_test, y_train, y_test = make_split()
    model = get_models()["logistic_regression"]
    model.fit(X_train, y_train)
    metrics = evaluate_test(model, X_test, y_test)

    bounded = [
        "accuracy", "balanced_accuracy", "precision", "sensitivity_recall",
        "specificity", "f1", "roc_auc", "average_precision",
    ]
    for name in bounded:
        assert 0.0 <= metrics[name] <= 1.0
    assert metrics["false_negatives"] >= 0
    assert metrics["false_positives"] >= 0


def test_cross_validation_returns_all_models():
    X_train, _, y_train, _ = make_split()
    result = cross_validate_models(get_models(), X_train, y_train)
    assert set(result["model"]) == set(get_models())
    assert result["roc_auc_mean"].notna().all()
