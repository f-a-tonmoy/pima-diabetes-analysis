"""Every number the report quotes, computed once from the data.

The report is generated, so no figure in its prose, stat band, captions or
tables is typed by hand. They are all read out of the dictionary this module
returns, and `tests/test_report.py` recomputes a sample of them independently
and checks that no number appears in the built HTML without a fact behind it.

Computing the set takes a couple of minutes (the model grid search and the
coefficient bootstrap dominate), so it is cached to `build/facts.json`. Pass
``refresh=True`` or delete the file to recompute.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve

from pima import cluster, eda, evaluate, explain
from pima.data import (
    FEATURES,
    IMPOSSIBLE_ZERO_COLUMNS,
    TARGET,
    load_clean,
    missingness_report,
    split,
)

CACHE = Path(__file__).resolve().parents[2] / "build" / "facts.json"


def _pct(x: float, places: int = 1) -> float:
    return round(100 * float(x), places)


def compute_facts() -> dict:
    """Run the analysis and collect every quantity the report cites."""
    df = load_clean()
    y_all = df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = split(df)

    f: dict = {}

    # ---- the cohort ------------------------------------------------------
    f["n_patients"] = int(len(df))
    f["n_features"] = int(len(FEATURES))
    f["n_diabetic"] = int(df[TARGET].sum())
    f["n_healthy"] = int((1 - df[TARGET]).sum())
    f["pct_diabetic"] = _pct(df[TARGET].mean())
    f["pct_healthy"] = _pct(1 - df[TARGET].mean())
    f["n_duplicates"] = int(df.duplicated().sum())

    # ---- missingness -----------------------------------------------------
    f["n_missing_total"] = int(df.isna().sum().sum())
    f["missing"] = {
        c: {"n": int(df[c].isna().sum()), "pct": _pct(df[c].isna().mean())}
        for c in IMPOSSIBLE_ZERO_COLUMNS
    }
    f["n_zero_pregnancies"] = int((df["Pregnancies"] == 0).sum())
    f["pct_zero_pregnancies"] = _pct((df["Pregnancies"] == 0).mean())

    overlap = eda.comissingness(df)
    f["comissing_skin_insulin"] = int(overlap.loc["SkinThickness", "Insulin"])
    f["comissing_bp_insulin"] = int(overlap.loc["BloodPressure", "Insulin"])
    f["comissingness"] = {
        "labels": list(overlap.columns),
        "matrix": overlap.to_numpy().tolist(),
    }

    miss = missingness_report(df)
    f["missingness_table"] = miss.to_dict("records")
    f["diabetes_rate_when_insulin_missing"] = float(
        miss.loc[miss["feature"] == "Insulin", "diabetes_rate_when_missing"].iloc[0]
    )
    f["overall_diabetes_rate"] = float(miss["overall_diabetes_rate"].iloc[0])

    # ---- distributions and group differences -----------------------------
    dist = eda.distribution_summary(df)
    f["distribution"] = dist.to_dict("records")
    f["insulin_skew"] = float(dist.loc[dist.feature == "Insulin", "skew"].iloc[0])
    f["insulin_max"] = float(dist.loc[dist.feature == "Insulin", "max"].iloc[0])
    f["insulin_sd"] = float(dist.loc[dist.feature == "Insulin", "sd"].iloc[0])
    f["dpf_max"] = float(dist.loc[dist.feature == "DiabetesPedigreeFunction", "max"].iloc[0])

    outliers = eda.outlier_summary(df)
    f["outliers"] = outliers.to_dict("records")
    f["n_outlier_rows"] = int(outliers.attrs["rows_with_any_outlier"])
    f["pct_outlier_rows"] = float(outliers.attrs["pct_rows_with_any_outlier"])

    groups = eda.group_comparison(df)
    f["group_comparison"] = groups.to_dict("records")
    f["cohens_d_glucose"] = float(groups.loc[groups.feature == "Glucose", "cohens_d"].iloc[0])
    f["cohens_d_min"] = float(groups["cohens_d"].abs().min())
    f["cohens_d_min_feature"] = str(groups.iloc[-1]["feature"])
    f["p_min"] = float(groups["mannwhitney_p"].min())
    f["p_max"] = float(groups["mannwhitney_p"].max())

    prevalence = {c: eda.binned_prevalence(df, c, bins=6) for c in FEATURES}
    f["prevalence"] = {c: v.to_dict("records") for c, v in prevalence.items()}
    gluc = prevalence["Glucose"]["diabetes_rate"]
    f["glucose_prevalence_low"] = _pct(gluc.iloc[0])
    f["glucose_prevalence_high"] = _pct(gluc.iloc[-1])
    f["glucose_prevalence_ratio"] = round(float(gluc.iloc[-1] / gluc.iloc[0]), 1)

    # ---- split -----------------------------------------------------------
    f["n_train"] = int(len(X_train))
    f["n_test"] = int(len(X_test))
    f["n_test_diabetic"] = int(y_test.sum())

    # ---- imputation ablation --------------------------------------------
    ablation = evaluate.imputation_ablation(
        X_train, y_train,
        LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"),
    )
    f["ablation"] = ablation.to_dict("records")
    f["ablation_spread"] = round(float(ablation.roc_auc.max() - ablation.roc_auc.min()), 4)
    f["ablation_sd"] = round(float(ablation.roc_auc_sd.mean()), 4)

    # ---- classification --------------------------------------------------
    best, board = evaluate.select_models(X_train, y_train)
    real = board[board.model != "majority_class"]
    f["leaderboard"] = board.drop(columns="best_params").to_dict("records")
    f["best_params"] = {r["model"]: r["best_params"] for r in board.to_dict("records")}
    f["cv_auc_min"] = float(real.roc_auc.min())
    f["cv_auc_max"] = float(real.roc_auc.max())
    f["cv_auc_range"] = round(float(real.roc_auc.max() - real.roc_auc.min()), 4)
    f["cv_auc_sd"] = round(float(real.roc_auc_sd.mean()), 3)
    f["cv_noise_ratio"] = int(round(f["cv_auc_sd"] / f["cv_auc_range"]))
    f["baseline_accuracy"] = _pct(board.loc[board.model == "majority_class", "accuracy"].iloc[0])
    f["tuned_accuracy_min"] = _pct(real.accuracy.min())
    f["tuned_accuracy_max"] = _pct(real.accuracy.max())
    f["accuracy_gain_over_baseline"] = round(
        _pct(real.accuracy.max()) - _pct(board.loc[board.model == "majority_class", "accuracy"].iloc[0]), 1
    )

    oof = {n: evaluate.out_of_fold_proba(m, X_train, y_train)
           for n, m in best.items() if n != "majority_class"}
    f["roc_curves"] = {}
    f["pr_curves"] = {}
    for name, proba in oof.items():
        fpr, tpr, _ = roc_curve(y_train, proba)
        step = max(1, len(fpr) // 160)
        f["roc_curves"][name] = {"fpr": fpr[::step].round(4).tolist(),
                                 "tpr": tpr[::step].round(4).tolist()}
        pr, rc, _ = precision_recall_curve(y_train, proba)
        step = max(1, len(pr) // 160)
        f["pr_curves"][name] = {"precision": pr[::step].round(4).tolist(),
                                "recall": rc[::step].round(4).tolist()}

    # ---- thresholds ------------------------------------------------------
    winner = board.loc[0, "model"]
    f["cv_winner"] = winner
    oof_best = oof[winner]
    rules = {r: evaluate.choose_threshold(y_train, oof_best, criterion=r)
             for r in ("f1", "youden", "min_recall", "cost")}
    f["thresholds"] = {r: evaluate.threshold_metrics(y_train, oof_best, t)
                       for r, t in rules.items()}
    f["threshold_min"] = round(min(rules.values()), 2)
    f["threshold_max"] = round(max(rules.values()), 2)
    cost, recall_rule = f["thresholds"]["cost"], f["thresholds"]["min_recall"]
    f["threshold_trade_cases"] = int(cost["tp"] - recall_rule["tp"])
    f["threshold_trade_alarms"] = int(cost["fp"] - recall_rule["fp"])

    pr, rc, thr = precision_recall_curve(y_train, oof_best)
    pr, rc = pr[:-1], rc[:-1]
    f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
    step = max(1, len(thr) // 200)
    f["threshold_sweep"] = {"threshold": thr[::step].round(4).tolist(),
                            "precision": pr[::step].round(4).tolist(),
                            "recall": rc[::step].round(4).tolist(),
                            "f1": f1[::step].round(4).tolist()}
    f["threshold_rules"] = {k: round(v, 4) for k, v in rules.items()}

    # ---- held-out test ---------------------------------------------------
    rows = []
    for name, model in best.items():
        own = evaluate.choose_threshold(
            y_train, evaluate.out_of_fold_proba(model, X_train, y_train), criterion="f1")
        proba = model.predict_proba(X_test)[:, 1]
        lo, hi = evaluate.bootstrap_ci(y_test, proba)
        rows.append({"model": name, **evaluate.threshold_metrics(y_test, proba, own),
                     "ci_low": lo, "ci_high": hi})
    f["test"] = rows
    test_real = [r for r in rows if r["model"] != "majority_class"]
    f["test_ci_width"] = round(float(np.mean([r["ci_high"] - r["ci_low"] for r in test_real])), 2)
    f["test_best_model"] = max(test_real, key=lambda r: r["roc_auc"])["model"]
    f["test_best_auc"] = max(r["roc_auc"] for r in test_real)
    lr = next(r for r in rows if r["model"] == "logistic_regression")
    f["final"] = lr
    f["final_recall_pct"] = _pct(lr["recall"])
    f["final_precision_pct"] = _pct(lr["precision"])
    f["cv_winner_test_rank"] = 1 + sorted(
        (r["roc_auc"] for r in test_real), reverse=True).index(
        next(r for r in test_real if r["model"] == winner)["roc_auc"])
    f["lr_cv_rank"] = 1 + list(real.model).index("logistic_regression")

    # ---- clustering ------------------------------------------------------
    matrix, imputed = cluster.prepare(df)
    ksel = cluster.k_selection(matrix)
    gsel = cluster.gmm_selection(matrix)
    f["k_selection"] = ksel.to_dict("records")
    f["gmm_selection"] = gsel.to_dict("records")
    f["silhouette_best"] = float(ksel.silhouette.max())
    f["k_by_silhouette"] = int(ksel.loc[ksel.silhouette.idxmax(), "k"])
    f["k_by_davies_bouldin"] = int(ksel.loc[ksel.davies_bouldin.idxmin(), "k"])
    f["k_by_bic"] = int(gsel.loc[gsel.bic.idxmin(), "k"])

    eps = cluster.suggest_dbscan_eps(matrix)
    algos = cluster.compare_clusterers(matrix, y_all, k=2, eps=eps)
    f["dbscan_eps"] = round(float(eps), 3)
    f["algorithms"] = algos.to_dict("records")
    f["best_ari"] = float(algos.adjusted_rand.max())
    f["best_ari_algorithm"] = str(algos.loc[algos.adjusted_rand.idxmax(), "algorithm"])
    f["dbscan_clusters"] = int(algos.loc[algos.algorithm == "dbscan", "n_clusters"].iloc[0])
    f["dbscan_noise"] = int(algos.loc[algos.algorithm == "dbscan", "n_noise"].iloc[0])
    f["ward_ari"] = float(algos.loc[algos.algorithm == "agglomerative_ward", "adjusted_rand"].iloc[0])

    scaling = cluster.scaling_comparison(df, y_all)
    f["scaling"] = scaling.to_dict("records")
    raw_row = scaling[scaling.features == "raw"].iloc[0]
    std_row = scaling[scaling.features == "standardised"].iloc[0]
    f["silhouette_raw"] = float(raw_row.silhouette)
    f["silhouette_std"] = float(std_row.silhouette)
    f["ari_raw"] = float(raw_row.adjusted_rand)
    f["ari_std"] = float(std_row.adjusted_rand)
    f["silhouette_ratio"] = round(f["silhouette_raw"] / f["silhouette_std"], 1)

    km = KMeans(n_clusters=2, n_init=10, random_state=65)
    f["stability"] = cluster.bootstrap_stability(matrix, km, n_resamples=50)
    labels = km.fit_predict(matrix)
    profile = cluster.cluster_profile(imputed, labels, y_all)
    f["cluster_profile"] = profile.round(2).to_dict("records")
    lo_i = int(profile.diabetes_rate.idxmin())
    hi_i = int(profile.diabetes_rate.idxmax())
    f["cluster_low"] = {"n": int(profile.loc[lo_i, "n"]), "rate": _pct(profile.loc[lo_i, "diabetes_rate"]),
                        "age": float(profile.loc[lo_i, "Age"]), "glucose": float(profile.loc[lo_i, "Glucose"]),
                        "bmi": float(profile.loc[lo_i, "BMI"]),
                        "pregnancies": float(profile.loc[lo_i, "Pregnancies"])}
    f["cluster_high"] = {"n": int(profile.loc[hi_i, "n"]), "rate": _pct(profile.loc[hi_i, "diabetes_rate"]),
                         "age": float(profile.loc[hi_i, "Age"]), "glucose": float(profile.loc[hi_i, "Glucose"]),
                         "bmi": float(profile.loc[hi_i, "BMI"]),
                         "pregnancies": float(profile.loc[hi_i, "Pregnancies"])}
    f["cluster_high_not_diabetic_pct"] = round(100 - f["cluster_high"]["rate"], 1)

    variance, loadings = cluster.pca_components(matrix)
    f["pca_variance"] = variance.to_dict("records")
    f["pca_cum2"] = _pct(variance.cumulative.iloc[1])
    f["pca_loadings"] = loadings.reset_index(names="component").to_dict("records")
    pca_xy = cluster.projections(matrix)["pca"]
    f["pca_points"] = {"x": pca_xy[:, 0].round(3).tolist(),
                       "y": pca_xy[:, 1].round(3).tolist(),
                       "cluster": labels.tolist(),
                       "outcome": y_all.tolist()}

    # ---- explainability --------------------------------------------------
    lr_model = evaluate.fit_tuned("logistic_regression", X_train, y_train)
    rf_model = evaluate.fit_tuned("random_forest", X_train, y_train)
    importances = {
        "logistic_regression": evaluate.permutation_importances(lr_model, X_test, y_test),
        "random_forest": evaluate.permutation_importances(rf_model, X_test, y_test),
    }
    comp = explain.compare_importances(importances)
    f["importances"] = comp.to_dict("records")
    row = comp[comp.feature == "Age"].iloc[0]
    f["age_importance_lr"] = float(row.logistic_regression)
    f["age_importance_rf"] = float(row.random_forest)
    f["age_importance_ratio"] = int(round(row.random_forest / max(row.logistic_regression, 1e-9)))
    preg = comp[comp.feature == "Pregnancies"].iloc[0]
    f["pregnancies_importance_lr"] = float(preg.logistic_regression)
    f["pregnancies_importance_rf"] = float(preg.random_forest)
    glu = comp[comp.feature == "Glucose"].iloc[0]
    f["glucose_importance_lr"] = float(glu.logistic_regression)
    f["glucose_importance_rf"] = float(glu.random_forest)

    coefs = explain.coefficient_table(lr_model, X_train, y_train, n_boot=400)
    f["coefficients"] = coefs.to_dict("records")
    f["n_coefficients"] = int(len(coefs))
    f["n_coefficients_nonzero"] = int(coefs.excludes_zero.sum())
    f["nonzero_features"] = [r["feature"] for r in coefs[coefs.excludes_zero].to_dict("records")]
    g = coefs[coefs.feature == "Glucose"].iloc[0]
    f["glucose_odds_ratio"] = float(g.odds_ratio)
    f["glucose_or_low"] = round(float(np.exp(g.ci_low)), 2)
    f["glucose_or_high"] = round(float(np.exp(g.ci_high)), 2)
    f["glucose_sd"] = round(float(X_train["Glucose"].fillna(X_train["Glucose"].median()).std()), 0)

    f["pdp"] = {}
    for feat in FEATURES:
        for tag, model in (("logistic_regression", lr_model), ("random_forest", rf_model)):
            frame = explain.partial_dependence_frame(model, X_train, feat, grid_resolution=30)
            f["pdp"].setdefault(feat, {})[tag] = {
                "x": frame[feat].round(3).tolist(),
                "y": frame["predicted_probability"].round(4).tolist()}

    _, fidelity, rules_text = explain.surrogate_tree(rf_model, X_train, max_depth=3)
    f["surrogate_fidelity"] = float(fidelity)
    f["surrogate_fidelity_pct"] = _pct(fidelity)
    f["surrogate_rules"] = rules_text
    f["surrogate_first_split"] = "Glucose" if "Glucose" in rules_text.split("\n")[0] else "?"

    proba_test = lr_model.predict_proba(X_test)[:, 1]
    threshold = evaluate.choose_threshold(
        y_train, evaluate.out_of_fold_proba(lr_model, X_train, y_train), criterion="f1")
    confident_idx = X_test.index[int(proba_test.argmax())]
    confident = explain.local_contributions(lr_model, X_test, confident_idx)
    f["patient_confident"] = {
        "id": int(confident_idx),
        "probability": confident.attrs["probability"],
        "contributions": confident.to_dict("records"),
        "top_feature": str(confident.iloc[0]["feature"]),
        "top_contribution": float(confident.iloc[0]["contribution"]),
        "rest_contribution": round(float(confident["contribution"][1:].sum()), 3),
    }
    missed = [i for i, (_, truth) in enumerate(y_test.items())
              if truth == 1 and proba_test[i] < threshold]
    missed_idx = X_test.index[missed[int(np.argmin(proba_test[missed]))]]
    miss_row = X_test.loc[missed_idx]
    f["patient_missed"] = {
        "id": int(missed_idx),
        "probability": explain.local_contributions(lr_model, X_test, missed_idx).attrs["probability"],
        "threshold": round(float(threshold), 2),
        "values": {c: (None if pd.isna(miss_row[c]) else float(miss_row[c])) for c in FEATURES},
        "cohort_means": {c: round(float(df[c].mean()), 1) for c in FEATURES},
    }
    return f


def load_facts(refresh: bool = False) -> dict:
    """Return the fact table, using the cache unless ``refresh`` is set."""
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    facts = compute_facts()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(facts, indent=1), encoding="utf-8")
    return facts


if __name__ == "__main__":
    import sys

    data = load_facts(refresh="--refresh" in sys.argv)
    print(f"{len(data)} top-level facts -> {CACHE}")
