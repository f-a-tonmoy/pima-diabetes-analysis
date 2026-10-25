import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pima import explain
from pima.data import FEATURES, TARGET, load_clean, split
from pima.models import build_pipeline, feature_names


@pytest.fixture(scope="module")
def fitted():
    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)
    model = build_pipeline(
        LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced")
    ).fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def test_local_contributions_reconstruct_the_prediction_exactly(fitted):
    """The decomposition is exact, not an approximation.

    This is the whole justification for not reaching for SHAP on the linear
    model. If contributions plus intercept ever stop reproducing
    predict_proba, the explanation is decorative and the notebook's claim is
    false.
    """
    model, _, X_test, _, _ = fitted
    for index in X_test.index[:25]:
        table = explain.local_contributions(model, X_test, index)
        assert table.attrs["probability"] == pytest.approx(
            table.attrs["model_probability"], abs=1e-9
        )


def test_local_contributions_cover_every_model_input(fitted):
    model, _, X_test, _, _ = fitted
    table = explain.local_contributions(model, X_test, X_test.index[0])
    assert set(table["feature"]) == set(feature_names(model))


def test_local_contribution_is_coefficient_times_value(fitted):
    model, _, X_test, _, _ = fitted
    table = explain.local_contributions(model, X_test, X_test.index[3])
    assert np.allclose(
        table["contribution"],
        (table["coefficient"] * table["value_standardised"]).round(3),
        atol=1e-3,
    )


def test_coefficient_table_intervals_bracket_the_point_estimate(fitted):
    model, X_train, _, y_train, _ = fitted
    table = explain.coefficient_table(model, X_train, y_train, n_boot=40)

    assert (table["ci_low"] <= table["coefficient"]).all()
    assert (table["coefficient"] <= table["ci_high"]).all()
    assert len(table) == len(feature_names(model))


def test_coefficient_table_survives_resamples_missing_an_indicator(fitted):
    """A bootstrap draw can contain no missing values for a rare column.

    `Glucose` is absent in only ~4 of 614 training rows, so some resamples have
    none, `SimpleImputer` emits no indicator for it, and the coefficient vector
    comes back shorter. Aligning on position instead of name silently
    misattributes every coefficient after the gap.
    """
    model, X_train, _, y_train, _ = fitted
    table = explain.coefficient_table(model, X_train, y_train, n_boot=60)
    assert table["ci_low"].notna().all()
    assert table["ci_high"].notna().all()


def test_coefficient_table_flags_zero_crossing_correctly(fitted):
    model, X_train, _, y_train, _ = fitted
    table = explain.coefficient_table(model, X_train, y_train, n_boot=40)
    expected = (table["ci_low"] > 0) | (table["ci_high"] < 0)
    assert (table["excludes_zero"] == expected).all()


def test_partial_dependence_frame_spans_the_feature(fitted):
    model, X_train, _, _, _ = fitted
    frame = explain.partial_dependence_frame(model, X_train, "Glucose", grid_resolution=20)

    assert list(frame.columns) == ["Glucose", "predicted_probability"]
    assert frame["predicted_probability"].between(0, 1).all()
    # a monotone model on a monotone feature must produce a rising curve
    assert frame["predicted_probability"].iloc[-1] > frame["predicted_probability"].iloc[0]


def test_compare_importances_aligns_on_feature():
    a = pd.DataFrame({"feature": ["x", "y"], "importance": [0.3, 0.1]})
    b = pd.DataFrame({"feature": ["y", "x"], "importance": [0.2, 0.4]})
    merged = explain.compare_importances({"model_a": a, "model_b": b}).set_index("feature")

    assert merged.loc["x", "model_a"] == 0.3
    assert merged.loc["x", "model_b"] == 0.4
    assert merged.loc["y", "model_b"] == 0.2


def test_surrogate_fidelity_is_measured_against_the_model_not_the_truth():
    """Fidelity must not be accuracy in disguise.

    A surrogate of a deliberately wrong model should score high fidelity (it
    mimics that model well) while being wrong about the labels. Conflating the
    two would make every surrogate look good.
    """
    df = load_clean()
    X_train, _, y_train, _ = split(df)
    inverted = build_pipeline(
        RandomForestClassifier(n_estimators=50, random_state=0)
    ).fit(X_train, 1 - y_train)

    _, fidelity, rules = explain.surrogate_tree(inverted, X_train, max_depth=3)
    agreement_with_truth = (inverted.predict(X_train) == y_train).mean()

    # the surrogate tracks the model it was fitted to...
    assert fidelity > 0.7
    # ...while that model is worse than chance about the actual labels, so a
    # fidelity score that were secretly accuracy would come out far lower
    assert agreement_with_truth < 0.5
    assert fidelity > agreement_with_truth + 0.25
    assert "Glucose" in rules
