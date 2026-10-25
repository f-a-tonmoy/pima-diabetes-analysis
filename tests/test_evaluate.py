import numpy as np
import pytest

from pima.evaluate import (
    bootstrap_ci,
    calibration_table,
    choose_threshold,
    threshold_metrics,
)


@pytest.fixture
def separable():
    """Scores that a threshold anywhere in (0.3, 0.7) classifies perfectly."""
    y = np.array([0] * 50 + [1] * 50)
    proba = np.concatenate([np.linspace(0.01, 0.3, 50), np.linspace(0.7, 0.99, 50)])
    return y, proba


def test_f1_threshold_separates_a_separable_problem(separable):
    y, proba = separable
    threshold = choose_threshold(y, proba, criterion="f1")
    metrics = threshold_metrics(y, proba, threshold)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["fn"] == 0


def test_youden_threshold_separates_a_separable_problem(separable):
    y, proba = separable
    metrics = threshold_metrics(y, proba, choose_threshold(y, proba, criterion="youden"))
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0


def test_min_recall_threshold_actually_reaches_the_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    proba = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.2, 400), 0, 1)

    for target in (0.7, 0.8, 0.9):
        threshold = choose_threshold(y, proba, criterion="min_recall", min_recall=target)
        assert threshold_metrics(y, proba, threshold)["recall"] >= target


def test_min_recall_raises_when_the_target_is_unreachable(separable):
    y, proba = separable
    with pytest.raises(ValueError, match="no threshold reaches"):
        choose_threshold(y, proba, criterion="min_recall", min_recall=1.01)


def test_cost_criterion_trades_false_alarms_for_missed_cases():
    """A heavier false-negative penalty must not raise the threshold."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    proba = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.25, 300), 0, 1)

    cheap = choose_threshold(y, proba, criterion="cost", fn_cost=1.0)
    expensive = choose_threshold(y, proba, criterion="cost", fn_cost=20.0)
    assert expensive <= cheap
    assert threshold_metrics(y, proba, expensive)["fn"] <= threshold_metrics(y, proba, cheap)["fn"]


def test_constant_scores_fall_back_to_the_nominal_threshold():
    y = np.array([0, 1, 0, 1])
    assert choose_threshold(y, np.full(4, 0.3), criterion="f1") == 0.5


def test_threshold_metrics_counts_the_confusion_matrix_correctly():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8])
    m = threshold_metrics(y, proba, 0.5)

    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5
    assert m["recall"] == 0.5
    assert m["specificity"] == 0.5


def test_unknown_criterion_is_rejected(separable):
    y, proba = separable
    with pytest.raises(ValueError, match="unknown criterion"):
        choose_threshold(y, proba, criterion="vibes")


def test_bootstrap_interval_brackets_the_point_estimate():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    proba = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.2, 300), 0, 1)

    lo, hi = bootstrap_ci(y, proba, n_resamples=400)
    assert lo < roc_auc_score(y, proba) < hi


def test_calibration_table_rows_sum_to_the_sample():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    proba = rng.uniform(size=200)

    table = calibration_table(y, proba, n_bins=5)
    assert table["n"].sum() == 200
    assert (table["observed"].between(0, 1)).all()
