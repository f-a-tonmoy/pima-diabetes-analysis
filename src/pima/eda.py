"""Exploratory analysis: distribution shape, outliers, and group differences.

Everything here is descriptive and every function returns a DataFrame, so the
notebook can show a table beside each figure rather than asking the reader to
take a chart's word for it.

Two habits shape this module. Group differences are reported as effect sizes
alongside p-values, because with 768 patients almost anything reaches
significance, and "p < 0.001" only says a difference is detectable. Outliers are
counted and never removed, because in clinical data an extreme insulin reading
is usually a real patient and the IQR rule flags skew as readily as it flags
errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pima.data import FEATURES, TARGET


def distribution_summary(df: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """Centre, spread and shape for each feature.

    Skew and the median/mean gap are the useful columns. A feature whose mean
    sits well above its median has a long right tail, which makes the median the
    safer choice for imputation and makes the raw scale dangerous for any
    distance-based method.
    """
    rows = []
    for col in features:
        values = df[col].dropna()
        rows.append(
            {
                "feature": col,
                "n": int(values.size),
                "mean": round(float(values.mean()), 2),
                "median": round(float(values.median()), 2),
                "sd": round(float(values.std()), 2),
                "min": round(float(values.min()), 2),
                "max": round(float(values.max()), 2),
                "skew": round(float(values.skew()), 2),
                "excess_kurtosis": round(float(values.kurtosis()), 2),
            }
        )
    return pd.DataFrame(rows)


def outlier_summary(df: pd.DataFrame, features=FEATURES, whis: float = 1.5) -> pd.DataFrame:
    """Count values outside the Tukey fences, and say what they would cost to drop.

    The last column matters most. Removing every flagged row is a common reflex,
    and on this dataset it would discard 84 patients, 10.9% of the cohort, most
    of them flagged only for a high insulin reading in a right-skewed
    distribution.
    """
    rows = []
    flagged = pd.Series(False, index=df.index)
    for col in features:
        values = df[col]
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - whis * iqr, q3 + whis * iqr
        mask = ((values < low) | (values > high)).fillna(False)
        flagged |= mask
        rows.append(
            {
                "feature": col,
                "lower_fence": round(float(low), 2),
                "upper_fence": round(float(high), 2),
                "n_outliers": int(mask.sum()),
                "pct": round(100 * float(mask.mean()), 1),
            }
        )
    summary = pd.DataFrame(rows).sort_values("n_outliers", ascending=False)
    summary.attrs["rows_with_any_outlier"] = int(flagged.sum())
    summary.attrs["pct_rows_with_any_outlier"] = round(100 * float(flagged.mean()), 1)
    return summary.reset_index(drop=True)


def group_comparison(df: pd.DataFrame, features=FEATURES, target: str = TARGET) -> pd.DataFrame:
    """Per-feature difference between outcome groups, as effect size and rank test.

    Cohen's d is the standardised mean difference: 0.2 is small, 0.5 moderate, 0.8
    large. The Mann-Whitney U test is used in place of a t-test because several
    of these distributions are strongly skewed, and it compares distributions
    without assuming normality.
    """
    rows = []
    for col in features:
        a = df.loc[df[target] == 0, col].dropna()
        b = df.loc[df[target] == 1, col].dropna()
        pooled_sd = np.sqrt(((a.var() * (len(a) - 1)) + (b.var() * (len(b) - 1)))
                            / (len(a) + len(b) - 2))
        statistic, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "feature": col,
                "mean_no_diabetes": round(float(a.mean()), 2),
                "mean_diabetes": round(float(b.mean()), 2),
                "cohens_d": round(float((b.mean() - a.mean()) / pooled_sd), 3),
                # rank-biserial correlation: the U statistic rescaled to [-1, 1]
                "rank_biserial": round(float(2 * statistic / (len(a) * len(b)) - 1), 3),
                "mannwhitney_p": float(f"{p:.2e}"),
            }
        )
    return (
        pd.DataFrame(rows)
        .assign(abs_d=lambda d: d["cohens_d"].abs())
        .sort_values("abs_d", ascending=False)
        .drop(columns="abs_d")
        .reset_index(drop=True)
    )


def comissingness(df: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """Which features go missing together.

    If `Insulin` and `SkinThickness` are almost always absent as a pair, they
    describe one event rather than two independent gaps: a patient who did not
    get a particular workup. That changes how much independent information
    imputation can recover.
    """
    missing = df[list(features)].isna()
    present = [c for c in features if missing[c].any()]
    return (
        missing[present].astype(int).T @ missing[present].astype(int)
    ).astype(int)


def binned_prevalence(
    df: pd.DataFrame, feature: str, bins: int = 6, target: str = TARGET
) -> pd.DataFrame:
    """Diabetes rate across bins of one feature.

    The direct answer to whether a variable relates to the outcome, and how. A
    monotone rise is a usable linear signal. A flat line means the feature
    carries nothing by itself, whatever its correlation coefficient says.
    """
    values = df[feature]
    binned = pd.qcut(values, bins, duplicates="drop")
    grouped = df.groupby(binned, observed=True)[target]
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "diabetes_rate": grouped.mean().round(3),
            "midpoint": [interval.mid for interval in grouped.size().index],
        }
    ).reset_index(names="bin")
    out["bin"] = out["bin"].astype(str)
    return out
