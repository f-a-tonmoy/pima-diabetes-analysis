import numpy as np
import pandas as pd
import pytest

from pima.data import (
    FEATURES,
    TARGET,
    load_clean,
    load_raw,
    missingness_report,
    split,
    unmask_missing,
)


def test_unmask_only_touches_impossible_columns():
    """Pregnancies == 0 is a real measurement and must survive unmasking.

    This is the bug the rule invites: "replace zeros with NaN" applied one
    column too wide silently deletes every first-time mother in the cohort.
    """
    df = pd.DataFrame(
        {
            "Pregnancies": [0, 3],
            "Glucose": [0, 120],
            "BloodPressure": [70, 0],
            "SkinThickness": [20, 20],
            "Insulin": [0, 90],
            "BMI": [0.0, 31.0],
            "DiabetesPedigreeFunction": [0.5, 0.2],
            "Age": [25, 40],
            TARGET: [0, 1],
        }
    )
    out = unmask_missing(df)

    assert out["Pregnancies"].tolist() == [0, 3]
    assert np.isnan(out.loc[0, "Glucose"])
    assert np.isnan(out.loc[1, "BloodPressure"])
    assert np.isnan(out.loc[0, "BMI"])
    assert out.loc[1, "Glucose"] == 120


def test_unmask_does_not_mutate_input():
    df = pd.DataFrame({c: [0] for c in FEATURES} | {TARGET: [0]})
    unmask_missing(df)
    assert df["Glucose"].tolist() == [0]


def test_load_raw_rejects_unexpected_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Glucose,Outcome\n120,1\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        load_raw(bad)


def test_load_raw_rejects_non_binary_target(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(",".join(FEATURES + (TARGET,)) + "\n" + "1," * 8 + "2\n")
    with pytest.raises(ValueError, match="binary"):
        load_raw(bad)


def test_split_is_stratified_and_disjoint():
    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)

    assert len(X_train) + len(X_test) == len(df)
    assert not set(X_train.index) & set(X_test.index)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)
    assert list(X_train.columns) == list(FEATURES)
    assert TARGET not in X_train.columns


def test_split_is_deterministic():
    df = load_clean()
    first, *_ = split(df)
    second, *_ = split(df)
    assert first.index.equals(second.index)


def test_missingness_report_matches_the_frame():
    df = load_clean()
    report = missingness_report(df).set_index("feature")
    for col in FEATURES:
        assert report.loc[col, "n_missing"] == int(df[col].isna().sum())
