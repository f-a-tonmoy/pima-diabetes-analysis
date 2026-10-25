import numpy as np
import pandas as pd
import pytest

from pima import eda
from pima.data import FEATURES, TARGET, load_clean


@pytest.fixture(scope="module")
def df():
    return load_clean()


def test_distribution_summary_counts_only_observed_values(df):
    """`n` must exclude NaN, or every spread statistic below it is wrong."""
    summary = eda.distribution_summary(df).set_index("feature")
    assert summary.loc["Insulin", "n"] == int(df["Insulin"].notna().sum())
    assert summary.loc["Age", "n"] == len(df)


def test_distribution_summary_detects_skew():
    """A deliberately right-skewed column must report positive skew."""
    frame = pd.DataFrame({"x": np.concatenate([np.ones(100), [50.0, 80.0]])})
    assert eda.distribution_summary(frame, features=["x"])["skew"].iloc[0] > 1


def test_outlier_summary_fences_match_the_tukey_rule():
    values = pd.DataFrame({"x": list(range(1, 101))})
    row = eda.outlier_summary(values, features=["x"]).iloc[0]

    q1, q3 = values["x"].quantile(0.25), values["x"].quantile(0.75)
    assert row["lower_fence"] == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert row["upper_fence"] == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert row["n_outliers"] == 0  # a uniform range has none


def test_outlier_summary_counts_rows_not_values(df):
    """The row count must not be the sum of per-feature counts.

    One patient can be an outlier on three features. Summing the columns
    double-counts them, which is how a 10.9% outlier rate gets reported as 26%.
    """
    summary = eda.outlier_summary(df)
    assert summary.attrs["rows_with_any_outlier"] <= int(summary["n_outliers"].sum())
    assert summary.attrs["rows_with_any_outlier"] <= len(df)
    assert 0 < summary.attrs["pct_rows_with_any_outlier"] < 100


def test_group_comparison_effect_size_sign_and_scale():
    """Cohen's d is positive when the positive class has larger values."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "high_in_cases": np.concatenate([rng.normal(0, 1, 300), rng.normal(2, 1, 300)]),
        "low_in_cases": np.concatenate([rng.normal(0, 1, 300), rng.normal(-2, 1, 300)]),
        TARGET: [0] * 300 + [1] * 300,
    })
    result = eda.group_comparison(
        frame, features=["high_in_cases", "low_in_cases"]
    ).set_index("feature")

    assert result.loc["high_in_cases", "cohens_d"] == pytest.approx(2.0, abs=0.2)
    assert result.loc["low_in_cases", "cohens_d"] == pytest.approx(-2.0, abs=0.2)


def test_group_comparison_finds_no_effect_when_there_is_none():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x": rng.normal(0, 1, 600), TARGET: [0] * 300 + [1] * 300})
    row = eda.group_comparison(frame, features=["x"]).iloc[0]
    assert abs(row["cohens_d"]) < 0.2
    assert row["mannwhitney_p"] > 0.01


def test_comissingness_diagonal_is_the_per_column_count(df):
    overlap = eda.comissingness(df)
    for col in overlap.columns:
        assert overlap.loc[col, col] == int(df[col].isna().sum())


def test_comissingness_is_symmetric(df):
    overlap = eda.comissingness(df)
    assert (overlap.to_numpy() == overlap.to_numpy().T).all()


def test_binned_prevalence_partitions_the_cohort(df):
    binned = eda.binned_prevalence(df, "Glucose", bins=6)
    assert binned["n"].sum() == df["Glucose"].notna().sum()
    assert binned["diabetes_rate"].between(0, 1).all()


def test_binned_prevalence_is_monotone_for_glucose(df):
    """Glucose is the one feature whose relationship should rise cleanly.

    If this ever fails, the linear-model conclusion drawn from it is suspect.
    """
    rates = eda.binned_prevalence(df, "Glucose", bins=6)["diabetes_rate"]
    assert rates.is_monotonic_increasing
