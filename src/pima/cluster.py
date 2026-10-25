"""Unsupervised structure: does this cohort fall into natural groups, and are
those groups diabetes?

The labels are used here only to score clusterings after the fact, never to fit
them. The question is whether unsupervised structure recovers the clinical
outcome by itself, so the measure that matters is agreement with the labels
(adjusted Rand, adjusted mutual information). A confusion matrix read after
relabelling clusters to whichever assignment looks best would not answer it.

One warning runs through this module: internal scores such as silhouette
measure whether clusters are geometrically tidy, which is a different question
from whether they mean anything. A clustering can score well and correspond to
nothing clinical, and on raw unscaled data it usually does. Insulin has a range
in the hundreds and DiabetesPedigreeFunction a range under three, so Euclidean
distance is essentially Insulin distance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pima.data import FEATURES, RANDOM_STATE


def prepare(
    df: pd.DataFrame,
    imputer: str = "median",
    scale: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Impute and optionally standardise the feature matrix for clustering.

    Returns the matrix handed to the algorithms and, separately, the imputed
    values in their original clinical units. The second is what cluster
    profiles are reported in, since "mean BMI 35.1" is readable and "mean BMI
    0.83 standard deviations" is not.

    No missing-indicator columns here. In supervised learning they are extra
    signal. In a distance-based clustering they would add a binary axis that
    pulls every patient with an untaken insulin test toward the same corner of
    the space, and the resulting clusters would describe who got tested instead
    of who is ill.
    """
    X = df[list(FEATURES)]
    step = (
        IterativeImputer(random_state=random_state, max_iter=20)
        if imputer == "iterative"
        else SimpleImputer(strategy=imputer)
    )
    imputed = pd.DataFrame(step.fit_transform(X), columns=list(FEATURES), index=X.index)
    matrix = StandardScaler().fit_transform(imputed) if scale else imputed.to_numpy()
    return matrix, imputed


def internal_scores(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette, Calinski-Harabasz and Davies-Bouldin for one clustering.

    All three are undefined unless at least two clusters contain points, and
    all three are computed on whatever geometry ``X`` is in, which is why
    they are not comparable between raw and standardised feature matrices.
    """
    mask = labels >= 0  # DBSCAN marks noise as -1; it is not a cluster
    if len(np.unique(labels[mask])) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "silhouette": round(float(silhouette_score(X[mask], labels[mask])), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X[mask], labels[mask])), 1),
        "davies_bouldin": round(float(davies_bouldin_score(X[mask], labels[mask])), 4),
    }


def agreement(y: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """How much of the diabetes label a clustering recovers.

    Both measures are adjusted for chance, so 0 means no better than a random
    partition of the same shape. Raw accuracy after relabelling starts at 50%
    for free on a binary outcome.
    """
    return {
        "adjusted_rand": round(float(adjusted_rand_score(y, labels)), 4),
        "adjusted_mutual_info": round(float(adjusted_mutual_info_score(y, labels)), 4),
    }


def k_selection(
    X: np.ndarray, k_range=range(2, 11), random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Sweep k for k-means, reporting inertia alongside the internal scores.

    The elbow in inertia is the plot everyone draws and the least informative of
    the four. Inertia falls monotonically with k by definition, so the elbow is
    a judgement call about where a smooth curve stops being steep. Silhouette
    and Davies-Bouldin at least have an optimum.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        rows.append({"k": k, "inertia": round(float(km.inertia_), 1), **internal_scores(X, labels)})
    return pd.DataFrame(rows)


def gmm_selection(
    X: np.ndarray, k_range=range(2, 11), random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Sweep component count for a Gaussian mixture, scored by BIC and AIC.

    A mixture model can be compared across k by likelihood penalised for
    parameters, which is a principled answer to "how many groups" that k-means
    cannot give. If BIC keeps falling to the edge of the range, that is itself
    evidence the data is one continuous blob being chopped up rather than a
    small number of real components.
    """
    rows = []
    for k in k_range:
        gm = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=random_state)
        labels = gm.fit_predict(X)
        rows.append(
            {
                "k": k,
                "bic": round(float(gm.bic(X)), 1),
                "aic": round(float(gm.aic(X)), 1),
                **internal_scores(X, labels),
            }
        )
    return pd.DataFrame(rows)


def suggest_dbscan_eps(X: np.ndarray, min_samples: int = 10) -> float:
    """Knee of the sorted k-nearest-neighbour distance curve, the usual eps heuristic."""
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    kth = np.sort(distances[:, -1])
    # farthest point from the straight line joining the curve's endpoints
    x = np.arange(len(kth), dtype=float)
    line = np.linspace(kth[0], kth[-1], len(kth))
    return float(kth[int(np.argmax(np.abs(kth - line)))])


def clusterers(k: int = 2, eps: float = 1.5, random_state: int = RANDOM_STATE) -> dict:
    """The candidate clustering algorithms.

    Chosen to differ in what they assume. k-means wants equal-sized spheres, the
    Gaussian mixture allows elongated overlapping components with soft
    assignment, Ward agglomerative builds a hierarchy without a centroid model,
    and DBSCAN looks for dense regions and is free to call most of the cohort
    noise. If a real two-group structure existed, these four should broadly
    agree on it.
    """
    return {
        "kmeans": KMeans(n_clusters=k, n_init=10, random_state=random_state),
        "gaussian_mixture": GaussianMixture(
            n_components=k, covariance_type="full", n_init=5, random_state=random_state
        ),
        "agglomerative_ward": AgglomerativeClustering(n_clusters=k, linkage="ward"),
        "dbscan": DBSCAN(eps=eps, min_samples=10),
    }


def compare_clusterers(
    X: np.ndarray, y: np.ndarray, k: int = 2, eps: float = 1.5, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Run every algorithm on the same matrix and score geometry and agreement side by side."""
    rows = []
    for name, algo in clusterers(k, eps, random_state).items():
        labels = clone(algo).fit_predict(X)
        found = len(set(labels) - {-1})
        rows.append(
            {
                "algorithm": name,
                "n_clusters": found,
                "n_noise": int(np.sum(labels == -1)),
                "largest_cluster_pct": round(
                    100 * np.bincount(labels[labels >= 0]).max() / len(labels), 1
                )
                if found
                else np.nan,
                **internal_scores(X, labels),
                **agreement(y, labels),
            }
        )
    return pd.DataFrame(rows)


def scaling_comparison(
    df: pd.DataFrame, y: np.ndarray, k: int = 2, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """k-means on raw versus standardised features.

    This is the trap the analysis exists to expose. Raw features give a far better
    silhouette, which looks like a better clustering and is not. The score is
    high because Insulin's hundreds-wide range dominates the distance metric, so
    k-means ends up splitting the cohort on one variable. Compare the agreement
    columns: silhouette here is measured in two different geometries, so the two
    numbers are not on the same scale.
    """
    rows = []
    for scale in (False, True):
        X, _ = prepare(df, scale=scale, random_state=random_state)
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
        rows.append(
            {
                "features": "standardised" if scale else "raw",
                **internal_scores(X, labels),
                **agreement(y, labels),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_stability(
    X: np.ndarray,
    algorithm,
    n_resamples: int = 50,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """How reproducible is a clustering when the sample changes?

    Two bootstrap resamples are drawn, each clustered independently, and the two
    labelings compared by adjusted Rand on the patients that appear in both.
    Real structure survives resampling. A partition imposed on a continuous
    cloud moves every time the sample does.

    This catches something no internal score can. A clustering may have a
    respectable silhouette and still be arbitrary, because silhouette asks
    whether the boundary is clean and never whether it would be drawn in the
    same place again.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    scores = []
    for _ in range(n_resamples):
        idx_a, idx_b = rng.integers(0, n, n), rng.integers(0, n, n)
        labels_a = clone(algorithm).fit_predict(X[idx_a])
        labels_b = clone(algorithm).fit_predict(X[idx_b])

        # first label seen for each original row, then compare on the overlap
        map_a = dict(zip(idx_a[::-1], labels_a[::-1]))
        map_b = dict(zip(idx_b[::-1], labels_b[::-1]))
        shared = sorted(set(map_a) & set(map_b))
        if len(shared) < 2:
            continue
        scores.append(
            adjusted_rand_score([map_a[i] for i in shared], [map_b[i] for i in shared])
        )
    return {
        "mean_ari": round(float(np.mean(scores)), 4),
        "sd_ari": round(float(np.std(scores)), 4),
        "min_ari": round(float(np.min(scores)), 4),
    }


def cluster_profile(
    imputed: pd.DataFrame, labels: np.ndarray, y: np.ndarray
) -> pd.DataFrame:
    """Mean feature values per cluster in clinical units, plus diabetes prevalence.

    This step decides whether a clustering is worth anything. A cluster is only
    interesting if you can say what kind of patient is in it.
    """
    profile = (
        imputed.assign(cluster=labels)
        .groupby("cluster")
        .mean()
        .round(1)
    )
    counts = pd.Series(labels).value_counts().sort_index()
    profile.insert(0, "n", counts.values)
    profile["diabetes_rate"] = (
        pd.Series(np.asarray(y)).groupby(labels).mean().round(3).values
    )
    return profile.reset_index()


def projections(X: np.ndarray, random_state: int = RANDOM_STATE) -> dict[str, np.ndarray]:
    """Two-dimensional views of the feature space.

    PCA is a linear projection, so distances in it mean something and it is fair
    to say two groups overlap there. t-SNE preserves local neighbourhoods and
    distorts global distance, so the size of its blobs and the gaps between them
    are artefacts of perplexity. It is included as the standard diagnostic, with
    the caveat attached.
    """
    return {
        "pca": PCA(n_components=2, random_state=random_state).fit_transform(X),
        "tsne": TSNE(
            n_components=2, perplexity=30, init="pca", random_state=random_state
        ).fit_transform(X),
    }


def pca_components(
    X: np.ndarray, n_components: int = 8, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explained variance per component, and what each component is made of.

    The loadings are the more useful half. A first component that loads evenly and
    positively on every feature is an overall-magnitude axis, which usually
    means the features are all mild proxies for one another.
    """
    pca = PCA(n_components=n_components, random_state=random_state).fit(X)
    variance = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_.round(4),
            "cumulative": pca.explained_variance_ratio_.cumsum().round(4),
        }
    )
    loadings = pd.DataFrame(
        pca.components_.round(3),
        index=[f"PC{i + 1}" for i in range(n_components)],
        columns=list(FEATURES),
    )
    return variance, loadings
