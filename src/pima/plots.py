"""Figures for the analysis.

Three conventions hold throughout, so that every chart in the notebook reads as
part of one system:

* Categorical colour is assigned in fixed slot order and never cycled, so a
  series keeps its colour across figures. The palette is colour-vision-safe
  (worst adjacent CVD delta-E 9.1) and identity is never carried by colour
  alone: every multi-series figure also has a legend, and most have a table
  beside them in the notebook.
* One y-axis per plot. Quantities on different scales get their own panel
  instead of a second axis, because a twin axis lets the author decide which
  line looks higher.
* Grid and axes recede; the data is the only thing at full contrast.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

#: Fixed categorical slots. Index by position, never by "next unused colour".
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdbd6"
SURFACE = "#fcfcfb"

#: Diverging map for correlations: two hues, neutral (not white, not a hue) midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "pima_diverging", ["#2a78d6", "#eeedea", "#eb6834"]
)
#: Single-hue sequential map for magnitude.
SEQUENTIAL = LinearSegmentedColormap.from_list("pima_seq", ["#eeedea", "#2a78d6"])


def use_style() -> None:
    """Apply the shared look. Call once at the top of the notebook."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "text.color": INK,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "figure.dpi": 110,
            "savefig.bbox": "tight",
        }
    )


def missingness(report: pd.DataFrame):
    """Share missing per feature, with the diabetes rate among missing rows beside it.

    Two panels instead of two axes, since the quantities are a percentage and a
    rate and putting them on one scale would invite a meaningless comparison.
    """
    present = report[report["n_missing"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))

    axes[0].barh(present["feature"], present["pct_missing"], color=SERIES[0], height=0.6)
    axes[0].set_title("Missing after unmasking impossible zeros")
    axes[0].set_xlabel("% of cohort")
    axes[0].invert_yaxis()
    for name, pct in zip(present["feature"], present["pct_missing"]):
        axes[0].text(pct + 0.8, name, f"{pct}%", va="center", color=INK_SOFT, fontsize=9)

    base = report["overall_diabetes_rate"].iloc[0]
    axes[1].barh(
        present["feature"],
        present["diabetes_rate_when_missing"],
        color=SERIES[1],
        height=0.6,
    )
    axes[1].axvline(base, color=INK_SOFT, linestyle="--", linewidth=1.2)
    axes[1].set_title("Diabetes rate when that feature is missing")
    # the reference line is named in the axis label rather than annotated on the
    # plot, where it collided with the title
    axes[1].set_xlabel(f"rate  (dashed = cohort rate {base:.3f})")
    axes[1].invert_yaxis()

    fig.tight_layout()
    return fig


def feature_distributions(df: pd.DataFrame, features, target: str = "Outcome"):
    """Small multiples: each feature's distribution split by outcome."""
    n = len(features)
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(13, 2.7 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, features):
        for cls, colour, label in ((0, SERIES[0], "no diabetes"), (1, SERIES[1], "diabetes")):
            values = df.loc[df[target] == cls, col].dropna()
            ax.hist(
                values,
                bins=25,
                density=True,
                color=colour,
                alpha=0.55,
                label=label,
                edgecolor=SURFACE,
                linewidth=0.5,
            )
        ax.set_title(col)
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.set_visible(False)

    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame):
    """Pearson correlations. Diverging scale, symmetric about zero."""
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iat[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=8,
                color="#ffffff" if abs(value) > 0.55 else INK,
            )
    ax.grid(False)
    ax.set_title("Feature correlations")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    return fig


def roc_and_pr(curves: dict[str, tuple], base_rate: float):
    """ROC and precision-recall for several models, side by side.

    Both panels are needed. ROC is insensitive to class balance, which makes it a
    fair ranking measure and an optimistic-looking one. The PR curve shows what
    performance on the 35% minority class costs, against the horizontal line a
    random model would sit on.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for slot, (name, (y, proba)) in enumerate(curves.items()):
        colour = SERIES[slot % len(SERIES)]
        fpr, tpr, _ = roc_curve(y, proba)
        axes[0].plot(fpr, tpr, color=colour, label=name.replace("_", " "))
        precision, recall, _ = precision_recall_curve(y, proba)
        axes[1].plot(recall, precision, color=colour, label=name.replace("_", " "))

    axes[0].plot([0, 1], [0, 1], color=INK_SOFT, linestyle="--", linewidth=1)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate")
    axes[0].set_title("ROC")

    axes[1].axhline(base_rate, color=INK_SOFT, linestyle="--", linewidth=1)
    axes[1].text(0.02, base_rate + 0.02, "random", color=INK_SOFT, fontsize=9)
    axes[1].set(xlabel="recall", ylabel="precision")
    axes[1].set_title("Precision-recall")

    axes[0].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def calibration(tables: dict[str, pd.DataFrame]):
    """Predicted probability against observed rate, per model."""
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], color=INK_SOFT, linestyle="--", linewidth=1, label="perfect")
    for slot, (name, table) in enumerate(tables.items()):
        ax.plot(
            table["mean_predicted"],
            table["observed"],
            marker="o",
            markersize=6,
            color=SERIES[slot % len(SERIES)],
            label=name.replace("_", " "),
        )
    ax.set(xlabel="mean predicted probability", ylabel="observed diabetes rate")
    ax.set_title("Calibration")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def threshold_sweep(y, proba, marked: dict[str, float] | None = None):
    """Precision, recall and F1 across every threshold, with chosen operating points.

    All three quantities live on 0-1, so they can share one axis. The vertical
    lines mark where each selection rule landed, which shows why 0.5 is a choice
    and not a neutral default.
    """
    precision, recall, thr = precision_recall_curve(y, proba)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for values, label, colour in (
        (precision, "precision", SERIES[0]),
        (recall, "recall", SERIES[1]),
        (f1, "F1", SERIES[2]),
    ):
        ax.plot(thr, values, color=colour, label=label)

    # Rules landing on the same threshold share one line. The lines are named
    # in the legend rather than annotated in place: selection rules routinely
    # agree to within a hundredth, and on-plot labels then sit on top of each
    # other.
    grouped: dict[float, list[str]] = {}
    for label, value in (marked or {}).items():
        grouped.setdefault(round(value, 3), []).append(label)

    for value, labels in sorted(grouped.items()):
        ax.axvline(
            value, color=INK_SOFT, linestyle=":", linewidth=1.2,
            label=f"{' = '.join(labels)} @ {value:.2f}",
        )

    ax.set(xlabel="decision threshold", ylabel="score", ylim=(0, 1.05))
    ax.set_title("What the threshold buys and costs")
    ax.legend(loc="lower left", fontsize=9, ncols=2)
    fig.tight_layout()
    return fig


def confusion(y, proba, threshold: float, title: str):
    """Counts at one operating point, annotated with what each cell means clinically."""
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(np.asarray(y), (proba >= threshold).astype(int), labels=[0, 1])
    meaning = [["correct", "false alarm"], ["MISSED CASE", "caught"]]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(matrix, cmap=SEQUENTIAL)
    for i in range(2):
        for j in range(2):
            strong = matrix[i, j] > matrix.max() * 0.55
            ax.text(
                j, i, f"{matrix[i, j]}\n{meaning[i][j]}",
                ha="center", va="center", fontsize=10,
                color="#ffffff" if strong else INK,
            )
    ax.set_xticks([0, 1], ["predicted no", "predicted yes"])
    ax.set_yticks([0, 1], ["actually no", "actually yes"])
    ax.grid(False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def importance_bar(importances: pd.DataFrame, title: str):
    """Permutation importance with its spread across repeats."""
    data = importances.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(
        data["feature"], data["importance"], xerr=data["sd"],
        color=SERIES[0], height=0.62,
        error_kw={"ecolor": INK_SOFT, "elinewidth": 1, "capsize": 3},
    )
    ax.axvline(0, color=INK_SOFT, linewidth=1)
    ax.set_xlabel("drop in ROC AUC when shuffled")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def k_selection_curves(kmeans_scores: pd.DataFrame, gmm_scores: pd.DataFrame):
    """How many clusters? Four criteria, four panels, no shared scale.

    Inertia, silhouette, Davies-Bouldin and BIC are on unrelated scales, so they
    get their own panels. They are meant to be read together: if they do not
    agree on a k, that disagreement is informative.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    panels = (
        (axes[0], kmeans_scores, "k", "inertia", "Inertia (always falls)", SERIES[0]),
        (axes[1], kmeans_scores, "k", "silhouette", "Silhouette (higher better)", SERIES[1]),
        (axes[2], kmeans_scores, "k", "davies_bouldin", "Davies-Bouldin (lower better)", SERIES[2]),
        (axes[3], gmm_scores, "k", "bic", "Mixture BIC (lower better)", SERIES[3]),
    )
    for ax, frame, x, y, title, colour in panels:
        ax.plot(frame[x], frame[y], marker="o", markersize=6, color=colour)
        ax.set_title(title)
        ax.set_xlabel("k")
    fig.tight_layout()
    return fig


def projection(embedding: np.ndarray, cluster_labels, true_labels, method: str):
    """The same 2-D projection coloured by cluster, then by the actual outcome.

    Side by side, because the comparison is the question being asked. If the
    clusters corresponded to diabetes, the two panels would look alike.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    panels = (
        (axes[0], np.asarray(cluster_labels), "cluster", "found clusters"),
        (axes[1], np.asarray(true_labels), "outcome", "true diabetes status"),
    )
    for ax, labels, kind, title in panels:
        for slot, value in enumerate(sorted(set(labels))):
            mask = labels == value
            name = (
                f"{kind} {value}" if kind == "cluster"
                else ("diabetes" if value == 1 else "no diabetes")
            )
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                s=18, alpha=0.75, linewidths=0.4, edgecolors=SURFACE,
                color=SERIES[slot % len(SERIES)], label=name,
            )
        ax.set_title(f"{method}: {title}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def boxplots_by_outcome(df: pd.DataFrame, features, target: str = "Outcome"):
    """Boxplots split by outcome, one panel per feature.

    Complements the histograms: a box shows the quartiles and the tail extent
    explicitly, which is what matters for features as skewed as these.
    """
    n = len(features)
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(13, 3.1 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, features):
        groups = [df.loc[df[target] == cls, col].dropna() for cls in (0, 1)]
        box = ax.boxplot(
            groups, tick_labels=["no", "yes"], patch_artist=True, widths=0.55,
            medianprops={"color": INK, "linewidth": 1.4},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK_SOFT,
                        "markeredgecolor": "none", "alpha": 0.4},
        )
        for patch, colour in zip(box["boxes"], (SERIES[0], SERIES[1])):
            patch.set(facecolor=colour, alpha=0.55, edgecolor=colour)
        ax.set_title(col)
        ax.set_xlabel("diabetes")
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def outlier_counts(summary: pd.DataFrame):
    """How many values each feature puts outside the Tukey fences."""
    data = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(data["feature"], data["n_outliers"], color=SERIES[0], height=0.62)
    for name, count, pct in zip(data["feature"], data["n_outliers"], data["pct"]):
        if count:
            ax.text(count + 1.5, name, f"{count}  ({pct}%)", va="center",
                    color=INK_SOFT, fontsize=9)
    ax.set_xlabel("values outside 1.5 x IQR")
    ax.set_title("Outliers by feature")
    fig.tight_layout()
    return fig


def effect_sizes(comparison: pd.DataFrame):
    """Standardised mean difference per feature, with interpretation bands marked."""
    data = comparison.iloc[::-1]
    colours = [SERIES[1] if d > 0 else SERIES[0] for d in data["cohens_d"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(data["feature"], data["cohens_d"], color=colours, height=0.62)
    for edge, label in ((0.2, "small"), (0.5, "moderate"), (0.8, "large")):
        ax.axvline(edge, color=INK_SOFT, linestyle=":", linewidth=1)
        # y in axes fraction so the band labels sit along the bottom rather than
        # riding up into the title
        ax.text(edge, 0.02, f" {label}", color=INK_SOFT, fontsize=8,
                transform=ax.get_xaxis_transform(), va="bottom")
    ax.axvline(0, color=INK_SOFT, linewidth=1)
    ax.set_xlabel("Cohen's d  (positive = higher in diabetic patients)")
    ax.set_title("Standardised difference between outcome groups")
    fig.tight_layout()
    return fig


def prevalence_curves(frames: dict[str, pd.DataFrame], base_rate: float):
    """Diabetes rate across bins of each feature, against the cohort rate."""
    n = len(frames)
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(13, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (feature, frame) in zip(axes, frames.items()):
        ax.plot(frame["midpoint"], frame["diabetes_rate"], marker="o",
                markersize=6, color=SERIES[0])
        ax.axhline(base_rate, color=INK_SOFT, linestyle="--", linewidth=1)
        ax.set_title(feature)
        ax.set_ylim(0, 1)
    for ax in axes[n:]:
        ax.set_visible(False)

    axes[0].set_ylabel("diabetes rate")
    fig.suptitle(f"Diabetes rate by feature bin  (dashed = cohort rate {base_rate:.2f})",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    return fig


def comissingness_heatmap(matrix: pd.DataFrame):
    """How often each pair of features is missing on the same patient."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=SEQUENTIAL)
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=9)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value = matrix.iat[i, j]
            ax.text(j, i, f"{value}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if value > matrix.values.max() * 0.55 else INK)
    ax.grid(False)
    ax.set_title("Patients missing both features")
    fig.tight_layout()
    return fig


def pca_scree(variance: pd.DataFrame):
    """Variance explained per component and cumulatively - two panels, two scales."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].bar(variance["component"], variance["explained_variance_ratio"],
                color=SERIES[0], width=0.62)
    axes[0].set_title("Variance explained per component")
    axes[0].set_ylabel("ratio")

    axes[1].plot(variance["component"], variance["cumulative"], marker="o",
                 markersize=6, color=SERIES[1])
    axes[1].axhline(0.9, color=INK_SOFT, linestyle="--", linewidth=1)
    axes[1].text(0, 0.91, "90%", color=INK_SOFT, fontsize=9)
    axes[1].set_title("Cumulative")
    axes[1].set_ylim(0, 1.05)

    fig.tight_layout()
    return fig


def component_loadings(loadings: pd.DataFrame, n_components: int = 3):
    """What each principal component is built from."""
    data = loadings.iloc[:n_components]
    fig, axes = plt.subplots(1, n_components, figsize=(4.3 * n_components, 3.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, row) in zip(axes, data.iterrows()):
        colours = [SERIES[1] if v > 0 else SERIES[0] for v in row]
        ax.barh(row.index, row.values, color=colours, height=0.62)
        ax.axvline(0, color=INK_SOFT, linewidth=1)
        ax.set_title(name)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig


def partial_dependence_curves(frames: dict[str, pd.DataFrame], base_rate: float):
    """Learned effect shape for each feature, averaged over the other features."""
    n = len(frames)
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(13, 2.9 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (feature, frame) in zip(axes, frames.items()):
        ax.plot(frame[feature], frame["predicted_probability"], color=SERIES[0])
        ax.axhline(base_rate, color=INK_SOFT, linestyle="--", linewidth=1)
        ax.set_title(feature)
    for ax in axes[n:]:
        ax.set_visible(False)

    axes[0].set_ylabel("predicted probability")
    fig.suptitle(f"Partial dependence  (dashed = cohort rate {base_rate:.2f})",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    return fig


def local_contribution_bar(table: pd.DataFrame, title: str):
    """One patient's log-odds, broken into per-feature contributions.

    Signed, so the two colours mean "pushed toward diabetes" and "pushed away";
    the legend names them, since colour alone is never the only encoding.
    """
    data = table[table["contribution"].abs() > 0.001].iloc[::-1]
    colours = [SERIES[1] if v > 0 else SERIES[0] for v in data["contribution"]]

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(data) + 1.6))
    ax.barh(data["feature"], data["contribution"], color=colours, height=0.62)

    # value labels sit outside the bar end, so the axis needs asymmetric padding
    # or a short negative bar pushes its label into the tick labels
    low, high = data["contribution"].min(), data["contribution"].max()
    span = max(high - low, 1e-6)
    ax.set_xlim(min(low, 0) - 0.22 * span, max(high, 0) + 0.14 * span)

    for name, value in zip(data["feature"], data["contribution"]):
        offset = 0.015 * span if value >= 0 else -0.015 * span
        ax.text(value + offset, name, f"{value:+.2f}", va="center",
                ha="left" if value >= 0 else "right", color=INK_SOFT, fontsize=9)
    ax.axvline(0, color=INK_SOFT, linewidth=1)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[1]),
               plt.Rectangle((0, 0), 1, 1, color=SERIES[0])]
    ax.legend(handles, ["raises risk", "lowers risk"], fontsize=9, loc="lower right")
    ax.set_xlabel("contribution to log-odds")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def importance_comparison(comparison: pd.DataFrame):
    """Permutation importance from several models, grouped by feature."""
    models = [c for c in comparison.columns if c != "feature"]
    y = np.arange(len(comparison))
    height = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8.5, 0.52 * len(comparison) + 1.8))
    for slot, model in enumerate(models):
        ax.barh(y + slot * height, comparison[model], height=height,
                color=SERIES[slot % len(SERIES)], label=model.replace("_", " "))
    ax.set_yticks(y + height * (len(models) - 1) / 2, comparison["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_SOFT, linewidth=1)
    ax.set_xlabel("drop in ROC AUC when shuffled")
    ax.set_title("Do the models agree on what matters?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
