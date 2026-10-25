import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from pima.cluster import (
    agreement,
    bootstrap_stability,
    cluster_profile,
    internal_scores,
    prepare,
    projections,
)
from pima.data import FEATURES, TARGET, load_clean


def test_agreement_is_one_for_an_identical_partition():
    labels = np.array([0, 0, 1, 1, 0, 1])
    assert agreement(labels, labels)["adjusted_rand"] == 1.0


def test_agreement_is_invariant_to_cluster_relabelling():
    """Cluster 0/1 are arbitrary names; swapping them must not change the score."""
    y = np.array([0, 0, 1, 1])
    assert agreement(y, np.array([1, 1, 0, 0])) == agreement(y, np.array([0, 0, 1, 1]))


def test_agreement_is_about_zero_for_a_random_partition():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    assert abs(agreement(y, rng.integers(0, 2, 500))["adjusted_rand"]) < 0.05


def test_internal_scores_ignore_dbscan_noise():
    """DBSCAN's -1 is "not in any cluster", not a third cluster.

    Scoring it as a cluster would penalise DBSCAN for the scattered points it
    correctly declined to assign, and make its silhouette incomparable to the
    other algorithms'.
    """
    from sklearn.metrics import silhouette_score

    X, labels = make_blobs(n_samples=60, centers=2, cluster_std=0.6, random_state=0)
    noisy = labels.copy()
    noisy[:5] = -1
    kept = noisy >= 0

    assert internal_scores(X, noisy)["silhouette"] == pytest.approx(
        silhouette_score(X[kept], noisy[kept]), abs=1e-4
    )
    assert internal_scores(X, noisy)["silhouette"] != pytest.approx(
        silhouette_score(X, noisy), abs=1e-4
    )


def test_internal_scores_are_undefined_for_a_single_cluster():
    X, _ = make_blobs(n_samples=30, centers=1, random_state=0)
    scored = internal_scores(X, np.zeros(30, dtype=int))
    assert np.isnan(scored["silhouette"])


def test_prepare_imputes_and_standardises():
    df = load_clean()
    matrix, imputed = prepare(df, scale=True)

    assert not np.isnan(matrix).any()
    assert matrix.shape == (len(df), len(FEATURES))
    assert np.allclose(matrix.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(matrix.std(axis=0), 1, atol=1e-9)
    # the unscaled companion stays in clinical units
    assert imputed["Glucose"].mean() > 100


def test_prepare_without_scaling_keeps_original_units():
    df = load_clean()
    matrix, imputed = prepare(df, scale=False)
    assert np.allclose(matrix, imputed.to_numpy())


def test_stability_separates_real_structure_from_an_imposed_partition():
    """Well-separated blobs re-cluster the same way; uniform noise does not.

    This is the property the whole stability argument rests on, so it is worth
    pinning: if bootstrap_stability returned a high score for structureless
    data, the conclusion drawn from it in the analysis would be worthless.
    """
    blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    noise = np.random.default_rng(0).uniform(size=(300, 2))
    km = KMeans(n_clusters=3, n_init=10, random_state=0)

    structured = bootstrap_stability(blobs, km, n_resamples=15, random_state=0)
    unstructured = bootstrap_stability(noise, km, n_resamples=15, random_state=0)

    assert structured["mean_ari"] > 0.9
    assert unstructured["mean_ari"] < structured["mean_ari"]


def test_cluster_profile_accounts_for_every_patient():
    df = load_clean()
    matrix, imputed = prepare(df)
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(matrix)

    profile = cluster_profile(imputed, labels, df[TARGET].to_numpy())
    assert profile["n"].sum() == len(df)
    assert set(profile.columns) >= {"cluster", "n", "diabetes_rate", "Glucose"}
    assert profile["diabetes_rate"].between(0, 1).all()


@pytest.mark.slow
def test_projections_return_two_dimensions_per_patient():
    df = load_clean()
    matrix, _ = prepare(df)
    for embedding in projections(matrix).values():
        assert embedding.shape == (len(df), 2)
