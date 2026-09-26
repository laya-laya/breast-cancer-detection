# Machine Learning for Breast Cancer Diagnosis

A reproducible biomedical machine-learning portfolio project using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.


## Question?

Can quantitative morphology features extracted from digitized fine-needle-aspirate images distinguish malignant from benign breast tumors, and how do different model classes trade off discrimination and clinically important classification errors?

## Highlights

A reproducible biomedical classification pipeline using the Wisconsin breast-cancer dataset. I separated model selection from final evaluation using a stratified held-out test set and five-fold cross-validation, compared linear and nonlinear classifiers, explicitly treated malignant tumors as the positive class, evaluated sensitivity, specificity, ROC-AUC and precision-recall behavior, and used permutation importance for post-hoc interpretation while accounting for correlated features.

## Dataset

The WDBC dataset contains **569 samples and 30 real-valued morphology features** derived from digitized images of fine-needle aspirates of breast masses. Scikit-learn's packaged version uses `0 = malignant` and `1 = benign`; this project recodes the target so that **malignant = 1**. This makes sensitivity/recall and false-negative counts clinically intuitive.

Class counts after recoding:

| Class | Samples |
|---|---:|
| Benign | 357 |
| Malignant | 212 |

The dataset is loaded directly with `sklearn.datasets.load_breast_cancer`, so no patient-level data file is committed to this repository.

## Evaluation design

The workflow separates the held-out test set **before model comparison**. Candidate models are compared using stratified 5-fold cross-validation on the training partition only. The highest mean training-CV ROC-AUC determines the selected model. The held-out set is then used for final performance evaluation and post-hoc interpretation.

This distinction matters: the model is **not chosen because of its test-set score**.

## Models

Three deliberately different model families are compared:

1. **Regularized logistic regression** with standardized predictors and class balancing
2. **Random forest** with class balancing
3. **Histogram gradient boosting**

The linear model provides an interpretable baseline, while the tree-based models test whether nonlinear decision boundaries materially improve discrimination.

## Validated results

The repository was validated with a deterministic random seed (`42`). On the training partition, logistic regression achieved the highest mean cross-validated ROC-AUC and was therefore selected.

## Repository structure

```text
breast-cancer-ml/
├── .github/workflows/tests.yml
├── data/
│   └── README.md
├── figures/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── precision_recall_curves.png
│   └── roc_curves.png
├── notebooks/
│   └── 01_complete_breast_cancer_ml.ipynb
├── results/
│   ├── cross_validation.csv
│   ├── permutation_importance.csv
│   └── test_metrics.csv
├── src/breast_cancer_ml/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluation.py
│   ├── models.py
│   └── train.py
├── tests/
│   ├── test_data.py
│   └── test_models.py
├── LICENSE
├── pyproject.toml
└── README.md
```

## Limitations

This project deliberately avoids overstating what can be concluded from a benchmark dataset.

- only 569 observations
- curated benchmark rather than a prospective clinical cohort
- no independent external validation cohort
- no demographic or acquisition-site analysis
- no probability calibration or decision-curve analysis
- morphology-derived tabular variables rather than raw medical images
- correlated predictors complicate feature-importance interpretation

High discrimination on WDBC does **not** establish clinical utility.


## Data provenance

- Scikit-learn dataset documentation: `sklearn.datasets.load_breast_cancer`
- Original dataset: Wisconsin Diagnostic Breast Cancer dataset, UCI Machine Learning Repository

## Author

**Laya Parkavousi**  
## License

Source code is released under the MIT License. Dataset provenance and licensing are governed by the original dataset source and scikit-learn distribution.
