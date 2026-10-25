"""Cross-validation, threshold selection and test-set reporting.

Two rules shape this module.

First, nothing is selected on the test set. Hyperparameters and the decision
threshold are both chosen from out-of-fold predictions on the training half;
the test set is scored exactly once, at the end, with everything already fixed.

Second, accuracy is read against a baseline. The cohort is 65% non-diabetic, so
a model that predicts "no" for everyone scores 65%. Ranking quality (ROC AUC),
performance on the minority class (average precision) and probability quality
(Brier score) all say more, and a missed diabetic and a false alarm carry
different costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from pima.data import RANDOM_STATE
from pima.models import build_pipeline, classifiers

SCORING = ("roc_auc", "average_precision", "balanced_accuracy", "accuracy")


def cv(n_splits: int = 5, n_repeats: int = 5, random_state: int = RANDOM_STATE):
    """Repeated stratified k-fold.

    Repeats matter here. A single 5-fold split of 614 rows gives validation
    folds of ~123, and the spread from one split to another is wide enough to
    reorder the leaderboard on its own.
    """
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )


def score_cv(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Mean and standard deviation of each metric over repeated CV."""
    res = cross_validate(
        estimator, X, y, scoring=list(SCORING), cv=cv(random_state=random_state)
    )
    out: dict[str, float] = {}
    for metric in SCORING:
        vals = res[f"test_{metric}"]
        out[metric] = round(float(vals.mean()), 4)
        out[f"{metric}_sd"] = round(float(vals.std()), 4)
    return out


def select_models(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "roc_auc",
    random_state: int = RANDOM_STATE,
    **preprocessor_kwargs,
) -> tuple[dict[str, BaseEstimator], pd.DataFrame]:
    """Grid-search every candidate and return the tuned best of each.

    Returns the fitted estimators keyed by name, and a leaderboard of
    cross-validated scores with standard deviations.
    """
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    best: dict[str, BaseEstimator] = {}
    rows = []

    for name, spec in classifiers(random_state).items():
        pipe = build_pipeline(
            clone(spec["estimator"]), random_state=random_state, **preprocessor_kwargs
        )
        search = GridSearchCV(pipe, spec["grid"], scoring=scoring, cv=inner, refit=True)
        search.fit(X, y)
        best[name] = search.best_estimator_

        rows.append(
            {
                "model": name,
                **score_cv(search.best_estimator_, X, y, random_state=random_state),
                "best_params": {
                    k.replace("model__", ""): v for k, v in search.best_params_.items()
                },
            }
        )

    leaderboard = (
        pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    )
    return best, leaderboard


def out_of_fold_proba(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Predicted probabilities for every training row, from a model that never saw it.

    These are what the decision threshold is chosen on. Picking a threshold from
    in-sample predictions, or from the test set, produces a number that will not
    hold up on new patients.
    """
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    return cross_val_predict(estimator, X, y, cv=folds, method="predict_proba")[:, 1]


def choose_threshold(
    y: np.ndarray,
    proba: np.ndarray,
    criterion: str = "f1",
    min_recall: float = 0.80,
    fn_cost: float = 5.0,
) -> float:
    """Pick a decision threshold under an explicit objective.

    ``f1``          balance precision and recall.
    ``youden``      maximise sensitivity + specificity - 1.
    ``min_recall``  most precise threshold that still catches ``min_recall`` of
                    diabetics (the screening framing).
    ``cost``        minimise ``fn_cost`` * false negatives + false positives.

    0.5 is not on this list. It is the default only because the library has to
    pick something, and it silently asserts that a missed diabetic costs the
    same as a false alarm.
    """
    y = np.asarray(y)
    if len(np.unique(proba)) < 2:
        # A constant scorer (the majority-class baseline) has no threshold that
        # separates anything. Return the nominal 0.5 rather than an arbitrary
        # point off a degenerate curve.
        return 0.5

    if criterion == "youden":
        fpr, tpr, thr = roc_curve(y, proba)
        return float(thr[np.argmax(tpr - fpr)])

    if criterion == "cost":
        candidates = np.unique(proba)
        costs = [
            fn_cost * np.sum((proba < t) & (y == 1)) + np.sum((proba >= t) & (y == 0))
            for t in candidates
        ]
        return float(candidates[int(np.argmin(costs))])

    precision, recall, thr = precision_recall_curve(y, proba)
    # precision_recall_curve returns one more curve point than thresholds
    precision, recall = precision[:-1], recall[:-1]

    if criterion == "f1":
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        return float(thr[np.argmax(f1)])

    if criterion == "min_recall":
        eligible = recall >= min_recall
        if not eligible.any():
            raise ValueError(f"no threshold reaches recall >= {min_recall}")
        idx = np.flatnonzero(eligible)[np.argmax(precision[eligible])]
        return float(thr[idx])

    raise ValueError(f"unknown criterion {criterion!r}")


def threshold_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Operating-point metrics at a given threshold, plus threshold-free scores."""
    y = np.asarray(y)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "avg_precision": round(float(average_precision_score(y, proba)), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
        "accuracy": round(float((tp + tn) / len(y)), 4),
        "recall": round(float(tp / (tp + fn)) if tp + fn else 0.0, 4),
        "precision": round(float(tp / (tp + fp)) if tp + fp else 0.0, 4),
        "specificity": round(float(tn / (tn + fp)) if tn + fp else 0.0, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def bootstrap_ci(
    y: np.ndarray,
    proba: np.ndarray,
    metric=roc_auc_score,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric on a held-out set.

    With 154 test rows the interval is wide, and that width matters: most of the
    gaps between these models sit comfortably inside it.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    n = len(y)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:  # metric undefined on a one-class resample
            continue
        stats.append(metric(y[idx], proba[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return round(float(lo), 4), round(float(hi), 4)


def calibration_table(
    y: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Predicted vs observed diabetes rate, by predicted-probability bin.

    A model can rank patients well and still be badly calibrated. If it says 70%
    for a group where 40% have diabetes, the ranking remains useful while the
    number does not.
    """
    bins = pd.qcut(proba, n_bins, labels=False, duplicates="drop")
    return (
        pd.DataFrame({"bin": bins, "proba": proba, "y": np.asarray(y)})
        .groupby("bin")
        .agg(n=("y", "size"), mean_predicted=("proba", "mean"), observed=("y", "mean"))
        .round(3)
        .reset_index(drop=True)
    )


def imputation_ablation(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Does the imputation strategy matter, and do missing-indicators earn their place?

    Half of Insulin is missing, so this is the preprocessing choice with the most
    room to change the answer, which makes it worth measuring.
    """
    rows = []
    for imputer in ("median", "mean", "iterative"):
        for add_indicator in (True, False):
            pipe = build_pipeline(
                clone(estimator),
                imputer=imputer,
                add_indicator=add_indicator,
                random_state=random_state,
            )
            scores = score_cv(pipe, X, y, random_state=random_state)
            rows.append(
                {
                    "imputer": imputer,
                    "missing_indicators": add_indicator,
                    "roc_auc": scores["roc_auc"],
                    "roc_auc_sd": scores["roc_auc_sd"],
                    "avg_precision": scores["average_precision"],
                }
            )
    return (
        pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    )


def permutation_importances(
    fitted_estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 30,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Drop in ROC AUC when each feature is shuffled.

    Model-agnostic, so a linear model and a forest can be compared on the same
    terms. Coefficients and impurity importances cannot: they are on different
    scales and measure different things.
    """
    res = permutation_importance(
        fitted_estimator,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": res.importances_mean.round(4),
                "sd": res.importances_std.round(4),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def fit_tuned(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "roc_auc",
    random_state: int = RANDOM_STATE,
    **preprocessor_kwargs,
) -> BaseEstimator:
    """Grid-search and fit a single named classifier.

    :func:`select_models` tunes all five, which suits the classification notebook
    and is more than a later notebook wanting a single model needs. This keeps
    the notebooks independent of each other, with no saved model files to go
    stale against the code that produced them.
    """
    spec = classifiers(random_state)[name]
    pipe = build_pipeline(
        clone(spec["estimator"]), random_state=random_state, **preprocessor_kwargs
    )
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = GridSearchCV(pipe, spec["grid"], scoring=scoring, cv=inner, refit=True)
    search.fit(X, y)
    return search.best_estimator_
