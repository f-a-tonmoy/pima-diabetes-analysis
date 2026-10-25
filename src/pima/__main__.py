"""Run the whole analysis headlessly: ``python -m pima``.

The notebook is the report; this is the same analysis without the prose, so
that the results can be regenerated and diffed without opening Jupyter, and so
CI has something to run.
"""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.cluster import KMeans

from pima import cluster, evaluate
from pima.data import TARGET, load_clean, missingness_report, split


def _section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _show(frame: pd.DataFrame) -> None:
    print(frame.to_string(index=False))


def main() -> int:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)
    print(f"{len(df)} patients, {df[TARGET].mean():.1%} diabetic; "
          f"train {len(X_train)} / test {len(X_test)}")

    _section("Missingness after unmasking impossible zeros")
    _show(missingness_report(df))

    _section("Cross-validated model selection (training half only)")
    best, leaderboard = evaluate.select_models(X_train, y_train)
    _show(leaderboard[["model", "roc_auc", "roc_auc_sd", "average_precision", "accuracy"]])

    winner = leaderboard.loc[0, "model"]
    _section(f"Threshold selection on out-of-fold predictions ({winner})")
    oof = evaluate.out_of_fold_proba(best[winner], X_train, y_train)
    thresholds = {
        rule: evaluate.choose_threshold(y_train, oof, criterion=rule)
        for rule in ("f1", "youden", "min_recall", "cost")
    }
    _show(pd.DataFrame(
        [{"rule": r, **evaluate.threshold_metrics(y_train, oof, t)}
         for r, t in thresholds.items()]
    ))

    _section("Held-out test set, scored once")
    rows = []
    for name, model in best.items():
        # each model gets the threshold its own out-of-fold predictions chose;
        # reusing one model's operating point for another compares nothing
        own_threshold = evaluate.choose_threshold(
            y_train, evaluate.out_of_fold_proba(model, X_train, y_train), criterion="f1"
        )
        proba = model.predict_proba(X_test)[:, 1]
        lo, hi = evaluate.bootstrap_ci(y_test, proba)
        rows.append({
            "model": name,
            **evaluate.threshold_metrics(y_test, proba, own_threshold),
            "roc_auc_95ci": f"[{lo}, {hi}]",
        })
    _show(pd.DataFrame(rows)[
        ["model", "roc_auc", "roc_auc_95ci", "avg_precision", "brier",
         "accuracy", "recall", "precision", "fn"]
    ])

    y_all = df[TARGET].to_numpy()
    matrix, imputed = cluster.prepare(df)

    _section("Clustering: how many groups, and are they diabetes?")
    _show(cluster.k_selection(matrix))
    eps = cluster.suggest_dbscan_eps(matrix)
    _show(cluster.compare_clusterers(matrix, y_all, k=2, eps=eps))

    _section("Scaling changes the answer (k-means, k=2)")
    _show(cluster.scaling_comparison(df, y_all))

    _section("Is the k=2 clustering reproducible, and what is in it?")
    km = KMeans(n_clusters=2, n_init=10, random_state=65)
    print(cluster.bootstrap_stability(matrix, km, n_resamples=30))
    _show(cluster.cluster_profile(imputed, km.fit_predict(matrix), y_all))

    return 0


if __name__ == "__main__":
    sys.exit(main())
