# Predicting diabetes in the Pima Indians cohort

Classification and clustering on the Pima Indians Diabetes dataset, with the
evaluation designed before the models were fitted.

The project asks two separate questions. Can eight routine clinical
measurements predict diabetes? And does this cohort divide into natural groups
that correspond to diabetes status? Both are answered below.

**[Read the interactive report](https://f-a-tonmoy.github.io/pima-diabetes-analysis/)** | [Notebooks](notebooks/)

Author: Fahim Ahamed.

## Findings

**Four model families score the same.** Logistic regression, an RBF SVM, a
400-tree random forest and gradient boosting span 0.8297 to 0.8352 ROC AUC in
repeated cross-validation, a range of 0.0055 against a standard deviation of
0.035. The flexible models found no non-linear structure that a regularised
linear model missed, which puts the limit on performance in the data itself.

**The test set cannot separate them either.** On 154 held-out patients the 95%
bootstrap interval for ROC AUC is about 0.12 wide, and every model's point
estimate falls inside every other model's interval. Logistic regression placed
third in cross-validation and first on the test set, which reflects the sample
it was scored on more than any difference between the models.

**The decision threshold matters more than the model choice.** Four selection
rules all landed between 0.39 and 0.45, well away from the 0.5 that `predict()`
uses. Moving between two of those rules trades 16 caught diabetics for 24 fewer
false alarms, which is a judgement for a clinician to make once the numbers are
in front of them.

**The missing values share a single cause.** Every patient missing
`SkinThickness` is also missing `Insulin`, as is every patient missing
`BloodPressure`. That nesting predicts what the ablation then confirms:
imputation strategy makes no measurable difference here (0.005 ROC AUC across
median, mean and iterative, against 0.033 of noise), and missing-value
indicators carry nothing.

**Every feature is statistically significant and only one is large.** All eight
clear p < 1e-5, but effect sizes run from `Glucose` at d = 1.19 down to
`BloodPressure` at 0.36. With 768 patients, significance measures sample size as
much as difference.

**The clusters are reproducible but do not correspond to diabetes.** No
algorithm (k-means, Gaussian mixture, Ward agglomerative, DBSCAN) recovered the
outcome better than adjusted Rand 0.16, and DBSCAN found a single dense region.
The k=2 split is nonetheless stable under bootstrap resampling (ARI 0.83) and
easy to interpret: younger, lower-BMI patients against older patients with
higher glucose, BMI and insulin. What the clustering picked up is a metabolic
risk gradient, with 17.5% diabetic in one cluster against 55.6% in the other.

**Equally accurate models rely on different features.** Logistic regression and
the random forest both put `Glucose` far ahead, but the forest weights `Age`
ten times more heavily, while the linear model routes that signal through the
correlated `Pregnancies`, which the forest ignores entirely. Only 4 of 13
bootstrap coefficient intervals exclude zero. One model's importances cannot be
treated as the explanation.

**One methodological result generalises past this dataset.** Clustering the
*raw*, unstandardised features scores a silhouette of 0.725 against 0.196 for
standardised ones, nearly four times better, while recovering essentially none
of the outcome (adjusted Rand 0.030 vs 0.161). `Insulin` ranges over hundreds
of units and `DiabetesPedigreeFunction` over less than three, so Euclidean
distance is almost entirely insulin distance. An internal validation score only
measures whether a boundary is geometrically clean.

## The report

**[Read the interactive report](https://f-a-tonmoy.github.io/pima-diabetes-analysis/)**

An eight-section write-up of the whole analysis with interactive charts, built
from the processed data so every number in it is generated rather than typed.
Source lives in `src/pima/report.py`; rebuild it with `python -m pima.report`,
which writes `docs/`, the directory GitHub Pages serves. The report is plain static
files, so there is nothing else to deploy.

## The notebooks

Read in order; each one summarises its results and hands off to the next. All
outputs are committed, so they read on GitHub without running anything.

| | Notebook | What it settles |
|---|---|---|
| 1 | [Data processing](notebooks/01_data_processing.ipynb) | 652 measurements hidden as zeros; which columns to unmask and which not to; the missingness is one nested event |
| 2 | [Exploratory analysis](notebooks/02_exploratory_analysis.ipynb) | Effect sizes vs p-values; why outliers are kept; the class overlap that sets the ceiling |
| 3 | [Classification](notebooks/03_classification.ipynb) | Leak-free pipelines, five models, threshold selection, one look at the test set |
| 4 | [Clustering](notebooks/04_clustering.ipynb) | Four algorithms, chance-adjusted agreement, bootstrap stability, PCA and t-SNE |
| 5 | [Explainability](notebooks/05_explainability.ipynb) | Permutation importance, bootstrap coefficients, partial dependence, surrogate rules, per-patient attribution |

The notebooks are thin. The logic lives in `src/pima/`, so what remains in them
is the argument and the figures.

## Running it

The analysis was developed in a conda environment (`intro_ds`, Python 3.12,
scikit-learn 1.8). Results are identical on scikit-learn 1.5.

```bash
conda create -n intro_ds python=3.12
conda activate intro_ds
pip install -e ".[dev]"
```

```bash
python -m pima          # the whole analysis, headless, ~2 minutes
python -m pima.report   # rebuild docs/index.html from the data
pytest -q               # 65 tests, including a claim audit of the report
jupyter notebook notebooks/
```

## How it is put together

```
src/pima/
  data.py       loading, unmasking disguised zeros, the train/test split
  eda.py        distribution shape, outliers, effect sizes, binned prevalence
  models.py     preprocessing pipelines and the classifier registry
  evaluate.py   cross-validation, threshold selection, bootstrap intervals
  cluster.py    clustering sweep, agreement metrics, bootstrap stability, PCA
  explain.py    permutation importance, coefficients, PDP, surrogate, local attribution
  plots.py      figures for the notebooks
  facts.py      every number the report quotes, computed once and cached
  figures.py    Plotly figures for the report
  report.py     builds docs/index.html
  __main__.py   `python -m pima`
tests/          unit tests per module, plus the report's claim audit
data/raw/       diabetes.csv, unmodified
data/processed/ written by notebook 1: unmasked only, never imputed
docs/           the built report, served by GitHub Pages
```

Three rules hold throughout, and the tests enforce the first:

1. **Nothing is fitted outside a pipeline.** Imputation and scaling are refit on
   every cross-validation fold. `test_models.py::test_preprocessor_is_fit_on_training_data_only`
   fails if anyone moves them out.
2. **Nothing is selected on the test set.** Hyperparameters and thresholds come
   from out-of-fold predictions on the training half. The test set is scored
   once.
3. **Accuracy is reported against a baseline.** 65% of the cohort is
   non-diabetic, so "always predict no" scores 65%. A `majority_class` entry
   sits in every leaderboard as the reference point.

The processed dataset is unmasked but not imputed. Unmasking a disguised zero
is a pure relabelling and is safe to do once, before any split. Imputation
learns from data, so it stays inside the pipeline. A `cleaned.csv` with the gaps
already filled is the artefact that makes leakage invisible three notebooks
later.

## Method

**Cleaning.** Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`
and `BMI` are biologically impossible and are unmasked to NaN. Zeros in
`Pregnancies` are real measurements and are left alone. A unit test pins that
distinction, because applying the rule one column too wide would silently delete
111 patients from the dataset's most complete feature.

**Classification.** Five candidates including a majority-class baseline, each
grid-searched in 5-fold CV on the training half, then re-scored with repeated
(5x5) stratified CV. Thresholds chosen from out-of-fold predictions under four
explicit objectives (F1, Youden, minimum-recall, cost-weighted). Test-set
metrics carry percentile bootstrap intervals.

**Clustering.** k selected by silhouette, Calinski-Harabasz, Davies-Bouldin and
mixture BIC across k=2..10, which settle on four different values. Four
algorithms with different assumptions, compared on chance-adjusted agreement
with the labels instead of relabelled accuracy. Bootstrap stability separates a
reproducible clustering from a meaningful one, which internal scores cannot do.
Clusters profiled in clinical units.

**Explainability.** SHAP is not used. For a linear model on standardised inputs
the additive decomposition is available in closed form: contributions plus
intercept reproduce `predict_proba` to floating-point error, which a test
asserts, so there is nothing for a sampled approximation to add. The ensemble is
covered by partial dependence and a surrogate tree reported with its fidelity
(88.3%), since rules from a low-fidelity surrogate describe nothing.

## Limitations

- **768 patients from one population**, women of Pima heritage aged 21+.
  Nothing here transfers elsewhere without revalidation.
- **154 test patients** give a bootstrap interval ~0.12 ROC AUC wide. Several
  conclusions above are limited by sample size rather than by modelling.
- **`Insulin` is 48.7% missing.** Its coefficient is partly a statement about
  the imputation.
- **`Glucose` is close to the diagnostic criterion** for diabetes, so its
  dominance is substantially definitional rather than a discovery.
- **Cross-sectional data.** Every association here is correlational.
- **Probabilities are not calibrated.** The tuned models use balanced class
  weights, which improves recall and distorts the probabilities. Treat the
  outputs as a ranking over patients, not as risk estimates.

## Data

`data/raw/diabetes.csv`: 768 rows, 8 features, binary `Outcome`. Originally
from the National Institute of Diabetes and Digestive and Kidney Diseases; all
patients are women of Pima heritage aged 21 or older. Redistributed unmodified.

## License

[MIT](LICENSE) for the code. The dataset is redistributed under its original terms.
