import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pima.data import FEATURES, load_clean, split
from pima.models import build_pipeline, build_preprocessor, classifiers, feature_names


def test_preprocessor_rejects_unknown_imputer():
    with pytest.raises(ValueError, match="imputer must be one of"):
        build_preprocessor(imputer="magic")


def test_preprocessing_leaves_no_missing_values():
    df = load_clean()
    prep = build_preprocessor()
    out = prep.fit_transform(df[list(FEATURES)])
    assert not np.isnan(out).any()


def test_missing_indicators_are_added_for_missing_columns_only():
    df = load_clean()
    prep = build_preprocessor(add_indicator=True).fit(df[list(FEATURES)])
    names = list(prep.get_feature_names_out())
    indicators = [n for n in names if n.startswith("missingindicator_")]

    # the five disguised-zero columns have missing values; the other three do not
    assert len(indicators) == 5
    assert "missingindicator_Insulin" in indicators
    assert "missingindicator_Age" not in indicators


def test_preprocessor_is_fit_on_training_data_only():
    """The leak check.

    A scaler fitted inside the pipeline must know only the training rows. If
    someone "helpfully" moves imputation or scaling outside the pipeline and
    fits it on the full frame, the fitted mean shifts to the full-data mean and
    this test fails.
    """
    df = load_clean()
    X_train, X_test, _, _ = split(df)

    pipe = build_pipeline(LogisticRegression(max_iter=5000)).fit(X_train, [0, 1] * (len(X_train) // 2))
    fitted_mean = pipe.named_steps["prep"].named_steps["scale"].mean_

    train_only = build_preprocessor().named_steps["impute"].fit_transform(X_train)
    full_data = build_preprocessor().named_steps["impute"].fit_transform(df[list(FEATURES)])

    assert np.allclose(fitted_mean, train_only.mean(axis=0))
    assert not np.allclose(fitted_mean, full_data.mean(axis=0))


def test_every_registered_classifier_fits_and_scores():
    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)

    for name, spec in classifiers().items():
        pipe = build_pipeline(spec["estimator"]).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        assert proba.shape == (len(X_test),), name
        assert ((proba >= 0) & (proba <= 1)).all(), name


def test_feature_names_cover_every_column_reaching_the_estimator():
    df = load_clean()
    X_train, _, y_train, _ = split(df)
    pipe = build_pipeline(LogisticRegression(max_iter=5000)).fit(X_train, y_train)

    names = feature_names(pipe)
    assert len(names) == pipe.named_steps["model"].coef_.shape[1]
    assert list(names)[: len(FEATURES)] == list(FEATURES)
