"""Preprocessing pipelines and the classifier registry.

Every estimator is wrapped in a Pipeline so that imputation and scaling are
refit inside each cross-validation fold. Fitting a scaler or an imputer on the
whole dataset before splitting is the most common way to leak test information
into a model, and it inflates every score that follows.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pima.data import RANDOM_STATE

IMPUTERS = ("median", "mean", "iterative")


def build_preprocessor(
    imputer: str = "median",
    add_indicator: bool = True,
    scale: bool = True,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Imputation followed by optional standardisation.

    ``add_indicator`` appends a binary "this value was missing" column for each
    feature that had missing values. Insulin is missing for almost half the
    cohort, and whether a measurement was taken at all is plausibly related to
    the clinical picture, so that fact is worth handing to the model instead of
    hiding it behind an imputed number.
    """
    if imputer not in IMPUTERS:
        raise ValueError(f"imputer must be one of {IMPUTERS}, got {imputer!r}")

    if imputer == "iterative":
        step = IterativeImputer(
            random_state=random_state, max_iter=20, add_indicator=add_indicator
        )
    else:
        step = SimpleImputer(strategy=imputer, add_indicator=add_indicator)

    steps = [("impute", step)]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def build_pipeline(estimator: BaseEstimator, **preprocessor_kwargs) -> Pipeline:
    """Wrap an estimator behind a fresh preprocessor."""
    return Pipeline(
        [
            ("prep", build_preprocessor(**preprocessor_kwargs)),
            ("model", estimator),
        ]
    )


def classifiers(random_state: int = RANDOM_STATE) -> dict[str, dict]:
    """The candidate classifiers and the grids searched over each.

    Grids are kept small. With 614 training rows the limit on this problem comes
    from the data rather than the hyperparameters, and a sprawling search mostly
    buys overfitting to the cross-validation folds.

    ``majority_class`` is the reference point rather than a competitor. 65% of
    this cohort is non-diabetic, so any accuracy figure has to be read against
    the 65% a model gets for free by predicting "no" every time.
    """
    return {
        "majority_class": {
            "estimator": DummyClassifier(strategy="most_frequent"),
            "grid": {},
        },
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=5000, random_state=random_state),
            "grid": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        "svm_rbf": {
            "estimator": SVC(probability=True, random_state=random_state),
            "grid": {
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", 0.01],
                "model__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=400, random_state=random_state
            ),
            "grid": {
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 5],
                "model__class_weight": [None, "balanced"],
            },
        },
        "gradient_boosting": {
            "estimator": HistGradientBoostingClassifier(
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=random_state,
            ),
            "grid": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [10, 20],
            },
        },
    }


def feature_names(fitted_pipeline: Pipeline) -> np.ndarray:
    """Column names as they reach the estimator, including missing-indicators."""
    return fitted_pipeline.named_steps["prep"].get_feature_names_out()
