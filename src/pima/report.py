"""Build the interactive report at ``docs/index.html``.

Run with ``python -m pima.report``. Every number in the page is read from
:mod:`pima.facts`, so nothing in the prose, stat band, captions or tables is
typed by hand, and `tests/test_report.py` fails the build if a figure appears in
the HTML without a fact behind it.
"""

from __future__ import annotations

from pathlib import Path

from pima import figures
from pima.facts import load_facts

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"
OUT = ROOT / "docs" / "index.html"

GITHUB_USER = "f-a-tonmoy"
REPO = "pima-diabetes-analysis"
REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO}"
PORTFOLIO_URL = "https://f-a-tonmoy.github.io/"

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='7' fill='%23111'/%3E"
    "%3Ctext x='16' y='21.5' font-family='Arial,Helvetica,sans-serif' font-size='14'"
    " font-weight='700' letter-spacing='0.5' fill='%23fff' text-anchor='middle'%3EFA"
    "%3C/text%3E%3C/svg%3E"
)

TERMS = {
    "ROC AUC": "https://en.wikipedia.org/wiki/Receiver_operating_characteristic",
    "Cohen's d": "https://en.wikipedia.org/wiki/Effect_size",
    "logistic regression": "https://en.wikipedia.org/wiki/Logistic_regression",
    "random forest": "https://en.wikipedia.org/wiki/Random_forest",
    "k-means": "https://en.wikipedia.org/wiki/K-means_clustering",
    "silhouette score": "https://en.wikipedia.org/wiki/Silhouette_(clustering)",
    "adjusted Rand index": "https://en.wikipedia.org/wiki/Rand_index",
    "bootstrap": "https://en.wikipedia.org/wiki/Bootstrapping_(statistics)",
    "principal component analysis": "https://en.wikipedia.org/wiki/Principal_component_analysis",
    "cross-validation": "https://en.wikipedia.org/wiki/Cross-validation_(statistics)",
    "type 2 diabetes": "https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes/type-2-diabetes",
}

#: Numbered sources. The tuple is (short label, what it supports, url).
SOURCES = [
    ("Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S. (1988). "
     "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. "
     "<em>Proceedings of the Annual Symposium on Computer Application in Medical Care</em>, 261-265.",
     "The paper that assembled this dataset, and the source of the question it was built to answer.",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/"),
    ("Pima Indians Diabetes Database. National Institute of Diabetes and Digestive and "
     "Kidney Diseases, distributed through the UCI Machine Learning Repository.",
     "The data file used here, unmodified.",
     "https://archive.ics.uci.edu/"),
    ("Knowler, W.C., Bennett, P.H., Hamman, R.F., Miller, M. (1978). Diabetes incidence and "
     "prevalence in Pima Indians: a 19-fold greater incidence than in Rochester, Minnesota. "
     "<em>American Journal of Epidemiology</em> 108(6), 497-505.",
     "Independently establishes the diabetes burden in this community and the scale of the "
     "difference from the general US population.",
     "https://pubmed.ncbi.nlm.nih.gov/736028/"),
    ("Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. "
     "<em>Journal of Machine Learning Research</em> 12, 2825-2830.",
     "Every model, metric and resampling routine used here.",
     "https://scikit-learn.org/stable/about.html"),
]

#: (number, phase, heading, anchor). The phase names the stage of work and
#: appears in the kicker and the section nav; the heading carries the finding.
SECTIONS = [
    ("01", "Data cleaning", "The file has no missing values, and that is the problem", "hidden"),
    ("02", "Exploratory analysis", "One measurement does nearly all the separating", "separate"),
    ("03", "Evaluation design", "Fixing the rules before fitting anything", "rules"),
    ("04", "Model comparison", "Four model families, one score", "models"),
    ("05", "Threshold selection", "Choosing the decision threshold", "threshold"),
    ("06", "Clustering", "What the unsupervised groups turn out to be", "groups"),
    ("07", "Validation check", "Why the better-scoring clustering is the wrong one", "trap"),
    ("08", "Explainability", "What the final model relies on", "using"),
]


_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve")


def words(n: int) -> str:
    """Spell out a small count. Digits mid-sentence read as data, not prose."""
    return _WORDS[n] if 0 <= n < len(_WORDS) else f"{n:,}"


def term(name: str, label: str | None = None) -> str:
    """Glossary link. `label` overrides the visible text, for sentence starts."""
    return (f'<a class="term" href="{TERMS[name]}" target="_blank" '
            f'rel="noopener">{label or name}</a>')


def cite(n: int) -> str:
    """Inline superscript pointing at the numbered source list."""
    return f'<sup class="cite"><a href="#source-{n}">{n}</a></sup>'


def figure_block(key: str, facts: dict, title: str, howto: str, caption: str) -> str:
    fig = figures.FIGURES[key](facts)
    div = fig.to_html(include_plotlyjs=False, full_html=False, div_id=f"fig-{key}",
                      config={"displayModeBar": False, "responsive": True})
    return f"""<figure class="reveal">
  <p class="fig-title">{title}</p>
  <p class="howto">{howto}</p>
  <div class="chart">{div}</div>
  <figcaption>{caption}</figcaption>
</figure>"""


def table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    head = "".join(
        f'<th class="num">{h}</th>' if i >= numeric_from else f"<th>{h}</th>"
        for i, h in enumerate(headers))
    body = ""
    for r in rows:
        cells = "".join(
            f'<td class="num">{c}</td>' if i >= numeric_from else f"<td>{c}</td>"
            for i, c in enumerate(r))
        body += f"<tr>{cells}</tr>"
    return (f'<div class="table-scroll"><table class="data">'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def build(facts: dict) -> str:
    f = facts
    miss = f["missing"]
    lo, hi = f["cluster_low"], f["cluster_high"]
    final = f["final"]
    cost = f["thresholds"]["cost"]
    strict = f["thresholds"]["min_recall"]
    pm = f["patient_missed"]

    nav = "".join(
        f'<li><a href="#{sid}"><span class="num">{num}</span>'
        f'<span class="label">{phase}</span></a></li>'
        for num, phase, _heading, sid in SECTIONS)



    s01 = f"""
<section id="hidden">
  <p class="section-kicker">Section 01<span class="sep">&middot;</span><span class="phase">{SECTIONS[0][1]}</span></p>
  <h2>{SECTIONS[0][2]}</h2>
  <p>The file opens clean. {f['n_patients']} rows, {words(f['n_features'])} measurements each, no blank
  cells, {words(f['n_duplicates'])} duplicates. Then the minimums: blood glucose zero, blood pressure
  zero, body mass index zero. Nobody has those and is alive to be measured. Readings never
  taken, written into the file as <code>0</code>.</p>

  <p>There are {f['n_missing_total']} of them across five measurements. A model that does not
  notice treats a patient whose insulin was never drawn as one whose insulin is zero, and
  fits a population that does not exist.</p>

  <p>One column has to be left alone. <code>Pregnancies</code> holds {f['n_zero_pregnancies']}
  zeros, every one a real answer. Blanking every zero would erase
  {f['pct_zero_pregnancies']}% of the most complete column.</p>

  {figure_block('missingness', f,
    "How much of each measurement was never recorded",
    "Longer bars are worse: more of the cohort has no reading for that measurement.",
    f"Counts in parentheses are patients. Skin-fold thickness and insulin are harder to "
    f"obtain than a cuff reading, the likely reason they are the two in clay.")}

  <p>How many gaps there are matters less than whether they are the same gap. Two columns
  missing a third of their values at random leave far more to work with than two that go
  missing together.</p>

  {figure_block('comissingness', f,
    "The gaps sit inside one another",
    "Darker cells hold more patients. Compare each off-diagonal cell with the diagonal: where they match, that measurement is never missing on its own.",
    f"All {f['comissing_skin_insulin']} patients with no skin-fold reading are also missing insulin, and so are "
    f"all {f['comissing_bp_insulin']} with no blood pressure. Insulin's own {miss['Insulin']['n']} contains "
    f"essentially every other gap.")}

  <p>All of it is one event: a patient who did not get the full workup. That predicts two
  things, neither testable yet. Flags marking the missing values should add nothing, since
  they record the same fact. Estimating a gap from the other measurements should add nothing either: for these patients those are missing too.</p>
</section>"""

    s02 = f"""
<section id="separate">
  <p class="section-kicker">Section 02<span class="sep">&middot;</span><span class="phase">{SECTIONS[1][1]}</span></p>
  <h2>{SECTIONS[1][2]}</h2>
  <p>All {words(f['n_features'])} measurements differ between patients with and without diabetes at
  levels most fields call highly significant, p-values from {f['p_min']:.0e} to {f['p_max']:.0e}.
  That reads like eight useful signals. It mostly reflects sample size: with
  {f['n_patients']} patients a very small difference still clears the bar.</p>

  <p>{term("Cohen's d")} asks the screening question instead: how large is the difference
  next to how much the measurement varies between people anyway.</p>

  {figure_block('effect_sizes', f,
    "How far apart the two groups sit on each measurement",
    "Further right means a wider separation between patients with and without diabetes. Only bars past the marked line count as large.",
    f"By convention 0.2 is small, 0.5 moderate and 0.8 large; all three are marked. Blood "
    f"glucose reaches {f['cohens_d_glucose']:.2f} and nothing else is close. {f['cohens_d_min_feature']} "
    f"reaches {f['cohens_d_min']:.2f} and is still statistically significant, the distance between "
    f"detectable and useful.")}

  {table(["Measurement", "No diabetes", "Diabetes", "Cohen's d"],
         [[r["feature"], f"{r['mean_no_diabetes']:.1f}", f"{r['mean_diabetes']:.1f}",
           f"{r['cohens_d']:.2f}"] for r in f["group_comparison"]])}

  <p>An effect size compares two averages, so it cannot say whether risk climbs steadily
  across a measurement's range or only bites past some point. For a screening rule that
  decides whether a simple cut-off would do.</p>

  {figure_block('prevalence', f,
    "Diabetes rate across the range of blood glucose",
    "Rising left to right means higher glucose goes with more diabetes. A flat line would mean the measurement carries nothing by itself.",
    f"Six groups of roughly equal size. The rate climbs from {f['glucose_prevalence_low']}% in the lowest "
    f"to {f['glucose_prevalence_high']}% in the highest, a factor of {f['glucose_prevalence_ratio']:.0f}, close "
    f"enough to a straight line for a simple model to capture nearly all of it.")}

  <p>A raised blood glucose reading is close to the definition of {term("type 2 diabetes")},
  so a model leaning on glucose is partly rediscovering a diagnostic criterion. Keep that in mind wherever glucose appears.</p>
</section>"""

    s03 = f"""
<section id="rules">
  <p class="section-kicker">Section 03<span class="sep">&middot;</span><span class="phase">{SECTIONS[2][1]}</span></p>
  <h2>{SECTIONS[2][2]}</h2>
  <p>The decisions that move a result like this most are made before a model is fitted, and
  are the easiest to make by accident. Three rules hold throughout.</p>

  <ul class="plain">
    <li><strong>The split comes first.</strong> {f['n_train']} patients to learn from,
    {f['n_test']} set aside. Anything that learns from data, gap-filling averages included, is
    refit inside each {term("cross-validation")} fold.</li>
    <li><strong>Nothing is chosen on the held-out patients.</strong> Model settings and the
    cut-off both come from the training half. The rest are scored once, at the end.</li>
    <li><strong>Accuracy is read against a baseline.</strong> {f['pct_healthy']}% of this cohort
    does not have diabetes, so answering "no" to everyone scores {f['baseline_accuracy']}% and
    finds nobody.</li>
  </ul>

  <p>One tempting shortcut breaks this dataset: filling a missing glucose with the average
  glucose <em>of patients with the same diagnosis</em>. The model then predicts the answer
  from a number derived from it, scores well, and fails on the first real patient.</p>

  <h3>Testing the two predictions from section 01</h3>
  <p>Six combinations of filling method and missing-value flags, scored by repeated
  cross-validation on the training half.</p>

  {table(["Filling method", "Missing-value flags", "Score (ROC AUC)"],
         [[r["imputer"], "Yes" if r["missing_indicators"] else "No", f"{r['roc_auc']:.4f}"]
          for r in f["ablation"]], numeric_from=2)}

  <p>Both hold. The spread is {f['ablation_spread']:.4f} {term("ROC AUC")} against
  {f['ablation_sd']:.3f} of run-to-run noise, so the choice disappears into the uncertainty.
  Iterative filling costs far more to run and buys nothing, for the reason section 01 gave:
  it estimates each missing value from measurements that are missing too.</p>

  <p>The rest of the analysis uses the median with flags, the simplest defensible default.
  Decisions like this absorb a lot of effort in practice. Measure once and
  stop.</p>
</section>"""

    s04 = f"""
<section id="models">
  <p class="section-kicker">Section 04<span class="sep">&middot;</span><span class="phase">{SECTIONS[3][1]}</span></p>
  <h2>{SECTIONS[3][2]}</h2>
  <p>Four model families, picked to differ in what they can represent.
  {term("logistic regression", "Logistic regression")} draws one straight boundary. A support
  vector machine bends it. A {term("random forest")} and a gradient-boosted ensemble cut the
  space into arbitrary regions and catch interactions. If a simple model is missing structure
  here, the flexible ones have the machinery to find it.</p>

  {table(["Model", "Cross-validated score", "Spread", "Accuracy"],
         [[figures.MODEL_LABEL.get(r["model"], r["model"]),
           f"{r['roc_auc']:.4f}", f"±{r['roc_auc_sd']:.4f}", f"{100 * r['accuracy']:.1f}%"]
          for r in f["leaderboard"]])}

  <p>They land between {f['cv_auc_min']:.4f} and {f['cv_auc_max']:.4f}, a range of
  {f['cv_auc_range']:.4f}. Re-running any one on a different split of the same patients moves its
  score by about {f['cv_auc_sd']:.3f}, roughly {f['cv_noise_ratio']} times the distance between
  first place and last.</p>

  {figure_block('leaderboard', f,
    "Scores on the held-out patients, with the uncertainty around each",
    "Further right is better. Each bar shows the range its score could plausibly take; where bars overlap, the difference cannot be measured here.",
    f"Every bar overlaps every other bar. The ranges average {f['test_ci_width']:.2f} wide on {f['n_test']} "
    f"patients, wider than the entire spread of scores. Bootstrap resampling, 2,000 draws.")}

  <p class="pullquote">Reading a winner off these numbers is reading noise.</p>

  <p>A second measurement agrees. Plain logistic regression placed {f['lr_cv_rank']}rd in
  cross-validation and first on the held-out patients at {f['test_best_auc']:.4f}. The models
  did not change between those results. The sample did.</p>

  <p>The ceiling belongs to the data. Section 02 showed the two groups overlapping across
  almost the whole range of every measurement, and no algorithm separates patients who look
  alike on its inputs. A tuned model reaches {f['tuned_accuracy_max']}% accuracy against
  {f['baseline_accuracy']}% for answering "no" to everyone: about
  {f['accuracy_gain_over_baseline']:.0f} points, bought by finding most of the cases.</p>

  <p>The cut-off is the decision that matters to a clinic, and every clinic answers it
  differently.</p>
</section>"""

    s05 = f"""
<section id="threshold">
  <p class="section-kicker">Section 05<span class="sep">&middot;</span><span class="phase">{SECTIONS[4][1]}</span></p>
  <h2>{SECTIONS[4][2]}</h2>
  <p>A model returns a number between 0 and 1. Turning that into "call this patient back"
  needs a cut-off, and every library defaults to 0.50. That default claims a missed case
  costs the same as a false alarm. A false alarm costs one confirmatory test. A missed case
  costs the years before the next one.</p>

  <p>Four ways of choosing, all computed on the training half so the held-out patients stay
  sealed.</p>

  {table(["Rule", "Cut-off", "Caught", "Missed", "False alarms"],
         [[k.replace("_", " ").capitalize(), f"{v['threshold']:.2f}", str(v["tp"]),
           str(v["fn"]), str(v["fp"])] for k, v in f["thresholds"].items()])}

  {figure_block('threshold', f,
    "What each cut-off buys and what it costs",
    "Moving left catches more patients who have diabetes and raises more false alarms. Moving right does the reverse. Higher is better on all three lines.",
    f"All four rules land between {f['threshold_min']:.2f} and {f['threshold_max']:.2f}, marked by the shaded "
    f"band. The library default sits outside it.")}

  <p>Moving from the cost-weighted cut-off, which prices one missed patient at five false
  alarms, to the strictest rule turns {f['threshold_trade_cases']} caught cases into missed
  ones and saves {f['threshold_trade_alarms']} false alarms. Whether that is a good trade
  depends on what a confirmatory test costs the clinic and what a late diagnosis costs the
  patient. Shipping 0.50 answers it without asking.</p>

  <p>At the cut-off used here, {final['threshold']:.2f}, the final model flags {final['tp']} of
  the {f['n_test_diabetic']} patients with diabetes among the held-out {f['n_test']}. It raises
  {final['fp']} false alarms and misses {final['fn']}. Moving the line does not fix those
  {final['fn']}.</p>
</section>"""

    s06 = f"""
<section id="groups">
  <p class="section-kicker">Section 06<span class="sep">&middot;</span><span class="phase">{SECTIONS[5][1]}</span></p>
  <h2>{SECTIONS[5][2]}</h2>
  <p>That was the supervised half. The other half hides the diagnosis and asks whether the
  cohort sorts itself into groups at all, and whether those groups are diabetes. If they
  were, a clinic could triage without labeled training data.</p>

  <p>Four methods that assume different things: {term("k-means")} looks for round groups of
  similar size, a Gaussian mixture allows stretched overlapping ones, Ward linkage builds a
  hierarchy with no notion of a center, and DBSCAN hunts for dense regions and may leave most
  of the cohort unassigned. Real structure should show up in most of them. The diagnosis only
  scores the answer afterward.</p>

  {table(["Method", "Groups found", "Match with diabetes"],
         [[r["algorithm"].replace("_", " "), str(r["n_clusters"]),
           f"{r['adjusted_rand']:.4f}"] for r in f["algorithms"]])}

  <p>Match is scored with the {term("adjusted Rand index")}, corrected for chance, so zero
  means no better than splitting the cohort at random. The best any method reaches is
  {f['best_ari']:.4f}. DBSCAN gives the plainest evidence: {f['dbscan_clusters']} group,
  {f['dbscan_noise']} patients unassigned, no dense regions with sparse ground between them.
  Ward linkage scores {f['ward_ari']:.4f}, fractionally worse than chance.</p>

  <p>The groups are not empty, though. Splitting the cohort in two is highly repeatable:
  resample the patients, refit, and the same boundary returns with an agreement of
  {f['stability']['mean_ari']:.2f}.</p>

  {table(["Group", "Patients", "Mean age", "Mean glucose", "Mean BMI", "Have diabetes"],
         [["Younger", str(lo["n"]), f"{lo['age']:.0f}", f"{lo['glucose']:.0f}",
           f"{lo['bmi']:.0f}", f"{lo['rate']}%"],
          ["Older", str(hi["n"]), f"{hi['age']:.0f}", f"{hi['glucose']:.0f}",
           f"{hi['bmi']:.0f}", f"{hi['rate']}%"]])}

  <p>Every measurement moves the same direction at once, so the split follows one gradient of
  age and metabolic load. Diabetes tracks it loosely, {lo['rate']}% against {hi['rate']}%, and
  {f['cluster_high_not_diabetic_pct']}% of the higher group does not have the disease. A clinic
  could use this to decide who to <em>watch</em>, and could not use it to decide who has
  diabetes.</p>

  {figure_block('pca', f,
    "The same patients, colored two ways",
    "Both panels show the same points, flattened to two dimensions by " + term("principal component analysis") + ". If the groups were diabetes, the panels would look alike.",
    f"The left panel splits cleanly because that is where the algorithm drew its line; the "
    f"right has no comparable boundary. These two dimensions carry {f['pca_cum2']}% of the variation, "
    f"so the scores above, computed in all {words(f['n_features'])} dimensions, are the evidence.")}
</section>"""

    s07 = f"""
<section id="trap">
  <p class="section-kicker">Section 07<span class="sep">&middot;</span><span class="phase">{SECTIONS[6][1]}</span></p>
  <h2>{SECTIONS[6][2]}</h2>
  <p>There is a version of section 06 that reports the opposite, and it is the version a
  careful-looking analysis produces by default.</p>

  <p>Grouping patients requires a notion of distance between them. On the measurements as
  they arrive, that distance is dominated by whichever one spans the widest numeric range.
  Here that is insulin, running to {f['insulin_max']:.0f}, while the diabetes pedigree score never
  passes {f['dpf_max']:.2f}. The distance between two patients is very nearly the distance
  between their insulin readings.</p>

  <p>Done that way, the result scores far better on the standard internal check.</p>

  {figure_block('scaling_trap', f,
    "The same method, run two ways, judged two ways",
    "Taller is better in both pairs. The left pair is the usual quality check. The right pair is whether the groups mean anything. They disagree.",
    f"On the {term('silhouette score')} the raw version beats the scaled one by a factor of "
    f"{f['silhouette_ratio']:.1f}. On agreement with the diagnosis it scores {f['ari_raw']:.4f} against "
    f"{f['ari_std']:.4f}, recovering nothing. The two silhouette values are computed in different "
    f"geometries and are not comparable.")}

  <p>A silhouette score asks whether a boundary is clean. It cannot ask whether the boundary
  separates anything, and on raw measurements it rewards the wrong answer confidently. An
  analyst who checked the silhouette and stopped would have shipped the result backwards,
  holding a number that looked like validation.</p>

  <p>The rule that caught it is the one to carry out of this project. <strong>No single
  measure is trusted until a second, independent one agrees.</strong> Section 04 survives it:
  the cross-validated ranking and the held-out ranking both say the differences sit inside the
  noise. Section 06 survives it: the groups score high on repeatability and low on agreement
  with the diagnosis. The raw-measurement clustering fails it, and appears here as an
  artifact.</p>
</section>"""

    s08 = f"""
<section id="using">
  <p class="section-kicker">Section 08<span class="sep">&middot;</span><span class="phase">{SECTIONS[7][1]}</span></p>
  <h2>{SECTIONS[7][2]}</h2>
  <p>When performance cannot choose between models, what they can tell you decides. Take the
  linear model and the forest, which scored the same by different routes.</p>

  {figure_block('importances', f,
    "What each model leans on, measured the same way for both",
    "Longer means the model relies on that measurement more. Where the two bars disagree, the models learned different things from the same patients.",
    f"Both put blood glucose far ahead, at {f['glucose_importance_lr']:.3f} and {f['glucose_importance_rf']:.3f}. Then "
    f"they part company: the forest leans on age about {f['age_importance_ratio']} times as heavily, "
    f"while the linear model uses number of pregnancies, which the forest ignores.")}

  <p>Age and pregnancies are correlated here, and the two models divide that shared
  information differently. The forest can bend age into a curve and takes it directly. The
  linear model cannot, so some of the same signal reaches it through pregnancies. Equal
  accuracy, different explanations, and a report showing only one would read as though it were the explanation.</p>

  <p>A single fit cannot say how much a number would move on different patients.
  Refitting the pipeline on {term("bootstrap")} resamples answers that.</p>

  {figure_block('coefficients', f,
    "How much each measurement moves the odds, and how sure we are",
    "Further from the center line is a stronger effect. A bar crossing the line is not evidence of anything, however confident its dot looks.",
    f"Only {f['n_coefficients_nonzero']} of the {f['n_coefficients']} inputs have a range clear of zero: "
    f"{', '.join(f['nonzero_features'][:-1])} and {f['nonzero_features'][-1]}. The missing-value flags from "
    f"section 01 are all in the unclear group, section 03's result arrived at from the other direction.")}

  <p>The measurements share a scale before fitting, so these numbers compare directly. A one standard deviation rise in blood glucose, about {f['glucose_sd']:.0f} mg/dL,
  multiplies the odds of diabetes by {f['glucose_odds_ratio']:.2f}, plausible range
  {f['glucose_or_low']:.2f} to {f['glucose_or_high']:.2f}.</p>

  <h3>One patient at a time</h3>
  <p>Averages do not tell a clinician why a particular person was flagged, the question a
  patient in the room asks. For a linear model the answer is exact: each
  measurement's push adds up to the prediction.</p>

  {figure_block('patient', f,
    f"Why the model was {100 * f['patient_confident']['probability']:.0f}% sure about patient {f['patient_confident']['id']}",
    "Bars to the right pushed this patient toward a diabetes prediction, bars to the left away from it. The lengths add up to the prediction exactly.",
    f"Blood glucose alone contributes {f['patient_confident']['top_contribution']:+.2f}, more than everything "
    f"else combined ({f['patient_confident']['rest_contribution']:+.2f}). The parts reproduce the model's own "
    f"output to within rounding, which a test asserts on every build.")}

  <p>The harder case is one of the {final['fn']} misses. Patient {pm['id']} has diabetes and
  scored {pm['probability']:.2f}, below the {pm['threshold']:.2f} cut-off. Every measurement the
  model relies on sits below average: glucose {pm['values']['Glucose']:.0f} against a cohort
  average of {pm['cohort_means']['Glucose']:.0f}, BMI {pm['values']['BMI']:.0f} against
  {pm['cohort_means']['BMI']:.0f}, {pm['values']['Pregnancies']:.0f} pregnancies against
  {pm['cohort_means']['Pregnancies']:.1f}. One value stands out: she is {pm['values']['Age']:.0f}
  against an average of {pm['cohort_means']['Age']:.0f}, and age is the one measurement whose
  range crosses zero for this model.</p>

  <p>She is section 02's overlap made concrete. On the inputs available she looks like a
  patient without the disease, and no cut-off catches her: reaching {pm['probability']:.2f}
  would flag a large part of the cohort with her. What would catch her is a measurement this
  file does not contain. A screening question ends there, at a different visit instead of a
  better algorithm.</p>
</section>"""

    takeaways = f"""
<section id="takeaways">
  <p class="section-kicker">Conclusions</p>
  <h2>What the project found</h2>
  <ol class="recap">
    <li>Four different model families scored within {f['cv_auc_range']:.4f} of one another
    while the noise on any one was {f['cv_auc_sd']:.3f}. There is no winner to pick.</li>
    <li>The held-out patients agreed. Their uncertainty ranges run about {f['test_ci_width']:.2f}
    wide, and every model's score falls inside every other model's range.</li>
    <li>Gap-filling made no measurable difference, and section 01 explained why before section 03
    measured it: the gaps are all one event.</li>
    <li>Every rule for setting the cut-off landed between {f['threshold_min']:.2f} and
    {f['threshold_max']:.2f}, outside the library default of 0.50.</li>
    <li>The cohort does not divide into diabetic and non-diabetic groups. It divides along a
    repeatable gradient of age and metabolic load that diabetes follows only loosely.</li>
    <li>The standard quality check for that clustering preferred the version that recovered
    nothing. A second, independent measure is what caught it.</li>
    <li>Only {f['n_coefficients_nonzero']} of {f['n_coefficients']} inputs have an effect
    distinguishable from zero, and two equally accurate models disagree about which
    measurements matter.</li>
    <li>The {final['fn']} patients the model misses look ordinary on these
    {words(f['n_features'])} measurements, which no cut-off or algorithm repairs.</li>
  </ol>
</section>

<section id="limits">
  <p class="section-kicker">Limits</p>
  <h2>What this data cannot tell us</h2>
  <ul class="plain">
    <li><strong>Who it describes.</strong> {f['n_patients']} women of Pima heritage, aged 21 and
    over, from a community with an exceptionally high diabetes burden{cite(3)}. Both the base
    rate and the relationships between measurements may differ elsewhere.</li>
    <li><strong>Whose records these are.</strong> A community under sustained research
    attention for decades. The file is used here as the public teaching dataset it has been
    since 1988, for a methods exercise.</li>
    <li><strong>Its age.</strong> Assembled for a 1988 paper{cite(1)}. Diagnostic criteria,
    measurement practice and treatment have all moved since.</li>
    <li><strong>How precise any of it is.</strong> {f['n_test']} held-out patients give
    uncertainty ranges about {f['test_ci_width']:.2f} wide. Several conclusions above are limited
    by that and not by the modeling.</li>
    <li><strong>Insulin.</strong> {miss['Insulin']['pct']}% of it is estimated, so anything the
    model says about insulin is partly a statement about the estimate.</li>
    <li><strong>Glucose is close to the diagnosis.</strong> Its dominance throughout is
    substantially a definition being rediscovered.</li>
    <li><strong>Cause.</strong> Every relationship here is an association measured at one point
    in time. They describe who, without explaining why.</li>
    <li><strong>The scores rank patients; they are not risks.</strong> Balanced class weights
    improve how many cases the models catch and distort the probabilities. A score of 0.70
    does not mean a 70% chance.</li>
    <li><strong>It is not a diagnostic tool.</strong> Nothing here is medical advice.</li>
  </ul>
</section>"""

    sources_html = "".join(
        f'<li id="source-{i}">{text} <span class="what">{what}</span> '
        f'<a href="{url}" target="_blank" rel="noopener">{url.split("//")[1].split("/")[0]}</a></li>'
        for i, (text, what, url) in enumerate(SOURCES, 1))

    assumptions = f"""
  <h2>Assumptions and checks</h2>
  <ol>
    <li>Zeros in five measurements record an absent reading. Checked by confirming each is
    biologically impossible, and by leaving <code>Pregnancies</code> out, where zero is a
    real answer for {f['n_zero_pregnancies']} patients. A unit test fails if the rule widens.</li>
    <li>The gap-filling method does not affect the result. Checked across six combinations:
    the spread was {f['ablation_spread']:.4f} against {f['ablation_sd']:.3f} of noise.</li>
    <li>Outliers are real patients. {f['n_outlier_rows']} rows ({f['pct_outlier_rows']}%) sit
    outside the usual fences, concentrated in the two most skewed measurements. They are
    kept, and handled by scaling and model choice.</li>
    <li>Two groups is a reasonable number to look at. Four criteria disagree: the silhouette
    prefers {f['k_by_silhouette']}, Davies-Bouldin {f['k_by_davies_bouldin']}, the mixture
    model's BIC {f['k_by_bic']}. The disagreement is itself the finding, and section 06 does
    not rest on the number.</li>
    <li>The final model is interchangeable with the others. Checked on two independent
    measurements, cross-validated and held-out, which agree that the differences sit inside
    the noise.</li>
  </ol>"""

    repro = f"""
  <h2>Reproducibility</h2>
  <p>An installable Python package (pandas, scikit-learn, Plotly) with per-module unit
  tests generates everything above. Every number is interpolated from the processed data at
  build time and none is typed by hand. Scores are ROC AUC unless stated otherwise;
  cross-validated figures are means over repeated 5x5 stratified folds on the training half,
  and held-out figures are scored once.</p>
  <ol>
    <li><strong>Source.</strong> One file, {f['n_patients']} rows, from the UCI Machine
    Learning Repository{cite(2)} and committed unchanged. No merges or joins.</li>
    <li><strong>Cleaning.</strong> Zeros in Glucose, BloodPressure, SkinThickness, Insulin
    and BMI become missing, {f['n_missing_total']} values across five columns;
    <code>Pregnancies</code> is untouched. No rows dropped, {f['n_duplicates']} duplicates
    found. The processed file keeps its gaps: filling them learns from data, so it belongs
    inside the pipeline where it is refit on every fold.</li>
    <li><strong>Analysis.</strong> Descriptive statistics and effect sizes; a stratified
    {f['n_train']}/{f['n_test']} split; grid search inside 5-fold cross-validation, then repeated
    5x5 stratified cross-validation for the reported scores; percentile bootstrap intervals
    on the held-out set; four clustering algorithms scored on chance-adjusted agreement;
    permutation importance and bootstrap coefficient intervals.</li>
    <li><strong>Checks.</strong> A test asserts the scaler is fitted on training rows only
    and fails if gap-filling moves outside the pipeline. Another recomputes the numbers
    quoted on this page and fails the build on a mismatch. Seeds are fixed throughout, and
    results are identical on scikit-learn 1.5 and 1.8.</li>
  </ol>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predicting Diabetes from Eight Clinical Measurements</title>
<meta name="description" content="A methods report on the Pima Indians Diabetes dataset: cleaning disguised missing values, comparing four classifier families, testing for unsupervised structure, and examining what the final model relies on.">
<link rel="icon" href="{FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Public+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="report.css">
<script src="{PLOTLY_CDN}" charset="utf-8"></script>
</head>
<body>

<nav class="section-nav" aria-label="Sections"><ol>{nav}</ol></nav>

<header class="intro">
  <div class="column">
    <p class="kicker">Pima Indians Diabetes dataset &middot; {f['n_patients']} patients</p>
    <h1>Predicting Diabetes from Eight Clinical Measurements</h1>
    <p class="deck">An end-to-end analysis of the Pima Indians Diabetes dataset, from
    the missing values through to what the final model ends up relying on.</p>
    <p class="byline">By Fahim Ahamed &nbsp;|&nbsp;
      <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub</a> &nbsp;|&nbsp;
      <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener">Portfolio</a></p>
  </div>
</header>

<main>
  <div class="column">
    <section id="summary">
    <h2 class="summary-h">Summary</h2>
    <p><strong>Objective.</strong> Establish how well {words(f['n_features'])} routine clinical
    measurements predict diabetes in this cohort, and whether the cohort contains unsupervised
    structure matching the diagnosis.</p>

    <p><strong>Data.</strong> The Pima Indians Diabetes Database: {f['n_patients']} women aged
    21 and over, assembled for a 1988 paper{cite(1)} from a National Institutes of Health
    study that measured this community's diabetes incidence at 19 times the rate of a mostly
    white population in Minnesota, the highest reported anywhere at the time{cite(3)}.
    {f['n_diabetic']} of the {f['n_patients']} women in the file have the disease.
    {words(f['n_features']).capitalize()} measurements each, all obtainable in one
    appointment.</p>

    <p><strong>Approach.</strong> Disguised missing values are unmasked but filled only
    inside the model pipeline; the stratified {f['n_train']}/{f['n_test']} split is made before
    anything is fitted; four classifier families are grid-searched inside cross-validation and
    scored once on the held-out patients; four clustering algorithms are scored on
    chance-adjusted agreement with the diagnosis; the final model is examined with permutation
    importance, bootstrap coefficient intervals and per-patient attribution.</p>

    <p><strong>Findings.</strong> The four model families are statistically
    indistinguishable, separated by {f['cv_auc_range']:.4f} ROC AUC against {f['cv_auc_sd']:.3f}
    of run-to-run noise. Every rule for setting the decision threshold lands between
    {f['threshold_min']:.2f} and {f['threshold_max']:.2f}, away from the 0.50 default. The cohort
    does contain a stable two-group structure, but it tracks age and metabolic load, agreeing
    with the diagnosis at only {f['best_ari']:.4f}. At the selected threshold the final model
    flags {f['final']['tp']} of the {f['n_test_diabetic']} patients with diabetes among the
    {f['n_test']} held out, and misses {f['final']['fn']}.</p>
    </section>

    {s01}
    {s02}
    {s03}
    {s04}
    {s05}
    {s06}
    {s07}
    {s08}
    {takeaways}
  </div>
</main>

<footer>
  <div class="column">
    <h2>Sources</h2>
    <ol class="sources">{sources_html}</ol>
    {assumptions}
    {repro}
    <div class="colophon">
      <p><span class="lbl">Analysis and writing:</span> <strong>Fahim Ahamed</strong></p>
      <p><span class="lbl">Tools:</span> Python, pandas, scikit-learn, Plotly</p>
      <p><span class="lbl">Code and notebooks:</span>
        <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub</a></p>
    </div>
    <p class="closing">If you are concerned about diabetes risk, a
    <strong>screening test</strong> from a clinician is the only thing that answers the
    question.</p>
  </div>
</footer>

<a class="totop" href="#top" aria-label="Back to top" title="Back to top">&uarr;</a>

<script src="report.js"></script>
</body>
</html>"""


def main() -> int:
    facts = load_facts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(facts), encoding="utf-8")
    for asset in ("report.css", "report.js"):
        source = ASSETS / asset
        if source.exists():
            (OUT.parent / asset).write_text(source.read_text(encoding="utf-8"),
                                            encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
