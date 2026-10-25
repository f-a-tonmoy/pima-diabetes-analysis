"""Explainability: what the models learned, globally and for one patient.

SHAP is not used here. For the linear model an exact decomposition is available
in closed form: each feature's contribution to the log-odds is its coefficient
times its standardised value, leaving nothing for a sampled approximation of the
same quantity to add. For the ensemble, partial dependence and a fitted
surrogate cover the questions that matter without another dependency.

If per-patient attributions for the tree models ever become a requirement, that
is the point to add `shap`. They are not one while the linear model is
statistically indistinguishable from the ensembles.

One caveat applies to everything in this module: these methods explain the
model and not the disease. They describe how a fitted function responds to its
inputs. Where features are correlated, and `Age`/`Pregnancies` and
`BMI`/`SkinThickness` are, a model can lean on one and ignore the other without
either being the real driver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import partial_dependence
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from pima.data import RANDOM_STATE
from pima.models import feature_names


def coefficient_table(
    fitted: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_boot: int = 400,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Standardised logistic coefficients with bootstrap confidence intervals.

    Point estimates alone invite over-reading. Refitting the whole pipeline on
    bootstrap resamples gives an interval for each coefficient, and a
    coefficient whose interval straddles zero is not evidence of anything,
    however confident its single-fit value looks.

    Coefficients are on standardised features, so they are comparable to one
    another: each is the change in log-odds per standard deviation. The same
    comparison is invalid on raw features, where a coefficient's size mostly
    reflects the unit a variable is measured in.
    """
    names = feature_names(fitted)
    point = fitted.named_steps["model"].coef_[0]

    rng = np.random.default_rng(random_state)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        refit = clone(fitted).fit(X.iloc[idx], y.iloc[idx])
        # Align on names, not position. A resample can happen to contain no
        # missing values for a rare column. Glucose is absent in only ~4 of
        # 614 training rows, and SimpleImputer then emits no indicator for it,
        # so the coefficient vector is shorter and positionally misaligned.
        draws.append(
            pd.Series(refit.named_steps["model"].coef_[0], index=feature_names(refit))
        )

    # columns missing from a draw stay NaN: that draw carries no information
    # about them, which is different from carrying a coefficient of zero
    drawn = pd.DataFrame(draws).reindex(columns=names)
    lo = np.nanquantile(drawn.to_numpy(), 0.025, axis=0)
    hi = np.nanquantile(drawn.to_numpy(), 0.975, axis=0)
    return (
        pd.DataFrame(
            {
                "feature": names,
                "coefficient": point.round(3),
                "ci_low": lo.round(3),
                "ci_high": hi.round(3),
                "odds_ratio": np.exp(point).round(3),
                "excludes_zero": (lo > 0) | (hi < 0),
            }
        )
        .sort_values("coefficient", key=abs, ascending=False)
        .reset_index(drop=True)
    )


def local_contributions(fitted: Pipeline, X: pd.DataFrame, index) -> pd.DataFrame:
    """Exact additive breakdown of one patient's predicted log-odds.

    For a linear model on standardised inputs this is not an approximation. The
    contributions plus the intercept sum to the log-odds exactly, which the
    caller can and should verify. Each row reads as "this patient's glucose is
    1.8 standard deviations above average, and that alone adds 1.9 to their
    log-odds".
    """
    row = X.loc[[index]]
    transformed = fitted.named_steps["prep"].transform(row)[0]
    coefficients = fitted.named_steps["model"].coef_[0]
    intercept = float(fitted.named_steps["model"].intercept_[0])
    contributions = coefficients * transformed

    table = pd.DataFrame(
        {
            "feature": feature_names(fitted),
            "value_standardised": transformed.round(3),
            "coefficient": coefficients.round(3),
            "contribution": contributions.round(3),
        }
    ).sort_values("contribution", key=abs, ascending=False).reset_index(drop=True)

    log_odds = intercept + contributions.sum()
    table.attrs["intercept"] = round(intercept, 3)
    table.attrs["log_odds"] = round(float(log_odds), 3)
    table.attrs["probability"] = round(float(1 / (1 + np.exp(-log_odds))), 3)
    table.attrs["model_probability"] = round(float(fitted.predict_proba(row)[0, 1]), 3)
    return table


def partial_dependence_frame(
    fitted: Pipeline, X: pd.DataFrame, feature: str, grid_resolution: int = 40
) -> pd.DataFrame:
    """Average predicted probability as one feature is swept across its range.

    Averaging over the observed distribution of the other features, so the curve
    answers what would happen if this patient's glucose were X and everything
    else stayed as it is. It shows the shape of a learned effect, which a single
    coefficient cannot: a plateau or a threshold looks nothing like a straight
    line.
    """
    position = list(X.columns).index(feature)
    result = partial_dependence(
        fitted, X, [position], grid_resolution=grid_resolution, kind="average"
    )
    return pd.DataFrame(
        {feature: result["grid_values"][0], "predicted_probability": result["average"][0]}
    )


def compare_importances(importances: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Line up permutation importances from several models side by side.

    Agreement is reassuring and disagreement is informative. If two models reach
    the same accuracy while relying on different features, neither has found the
    explanation, and the data supports more than one account.
    """
    merged = None
    for name, frame in importances.items():
        column = frame[["feature", "importance"]].rename(columns={"importance": name})
        merged = column if merged is None else merged.merge(column, on="feature")
    return merged.set_index("feature").round(4).reset_index()


def surrogate_tree(
    fitted: Pipeline,
    X: pd.DataFrame,
    max_depth: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[DecisionTreeClassifier, float, str]:
    """Fit a shallow tree to mimic a black-box model, and report how well it does.

    Returns the tree, its fidelity (agreement with the original model's
    predictions, not with the truth), and readable rules.

    Fidelity decides whether the rules are worth believing. A surrogate agreeing
    with the model 90% of the time is a fair summary of it. One agreeing 60% of
    the time is a different model whose rules describe nothing.
    """
    predictions = fitted.predict(X)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X.fillna(X.median()), predictions)

    fidelity = float((tree.predict(X.fillna(X.median())) == predictions).mean())
    rules = export_text(tree, feature_names=list(X.columns), decimals=1)
    return tree, round(fidelity, 4), rules
