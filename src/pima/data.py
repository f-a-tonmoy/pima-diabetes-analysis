"""Loading, cleaning and splitting the Pima Indians Diabetes dataset.

The only cleaning that happens here is unmasking: turning biologically
impossible zeros into explicit NaN. Imputation does not live in this module. It
belongs inside a scikit-learn pipeline so that it is refit on each training fold
and never sees the validation data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 65
TARGET = "Outcome"

#: A recorded zero in these columns is not a measurement of zero. A living
#: patient cannot have zero glucose, blood pressure, skin fold thickness,
#: serum insulin or BMI. They are missing values written as zero by whoever
#: assembled the dataset.
IMPOSSIBLE_ZERO_COLUMNS = (
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
)

FEATURES = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "diabetes.csv"


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the CSV exactly as distributed, with the expected schema enforced."""
    df = pd.read_csv(DATA_PATH if path is None else path)
    missing = set(FEATURES + (TARGET,)) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing expected columns: {sorted(missing)}")
    if not df[TARGET].isin((0, 1)).all():
        raise ValueError(f"{TARGET} must be binary 0/1")
    return df


def unmask_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zeros with NaN. Returns a copy; the input is untouched."""
    out = df.copy()
    out[list(IMPOSSIBLE_ZERO_COLUMNS)] = out[list(IMPOSSIBLE_ZERO_COLUMNS)].replace(
        0, np.nan
    )
    return out


def load_clean(path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset with disguised missing values made explicit."""
    return unmask_missing(load_raw(path))


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    The test set is set aside here and must not influence any modelling
    decision. Imputation, scaling and hyperparameter search all happen on the
    training half only.
    """
    X = df[list(FEATURES)]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing count and share, plus how missingness tracks the target.

    The last column is the diabetes rate among rows where the feature is missing.
    If it differs from the overall rate, missingness itself carries signal, and
    dropping or naively imputing those rows throws information away.
    """
    base_rate = df[TARGET].mean()
    rows = []
    for col in FEATURES:
        na = df[col].isna()
        rows.append(
            {
                "feature": col,
                "n_missing": int(na.sum()),
                "pct_missing": round(100 * na.mean(), 1),
                "diabetes_rate_when_missing": (
                    round(df.loc[na, TARGET].mean(), 3) if na.any() else np.nan
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .assign(overall_diabetes_rate=round(base_rate, 3))
        .sort_values("n_missing", ascending=False)
        .reset_index(drop=True)
    )


PROCESSED_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "diabetes_unmasked.csv"


def save_processed(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Write the unmasked dataset to ``data/processed/``.

    Only the unmasking is persisted, never imputed values. Unmasking is a pure
    relabelling: a zero becomes a NaN and nothing is estimated from other rows,
    so it is safe to do once, before any split. Imputation learns from data, so
    it belongs inside the pipeline where it is refit per fold. A "cleaned.csv"
    with the gaps already filled is the artefact that makes leakage invisible
    three notebooks later.
    """
    target = Path(PROCESSED_PATH if path is None else path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def load_processed(path: str | Path | None = None) -> pd.DataFrame:
    """Read the unmasked dataset written by :func:`save_processed`."""
    target = Path(PROCESSED_PATH if path is None else path)
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. Run notebooks/01_data_processing.ipynb first, "
            "or call load_clean() to derive it from the raw CSV."
        )
    return pd.read_csv(target)
