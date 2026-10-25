# Predicting diabetes in the Pima Indians cohort

Classification and clustering on the Pima Indians Diabetes dataset, with the
evaluation designed before any model was fitted.

**[Read the interactive report](https://f-a-tonmoy.github.io/pima-diabetes-analysis/)** | [Notebooks](notebooks/)

Two questions: can eight routine clinical measurements predict diabetes, and
does the cohort split into natural groups that match the diagnosis?

## Key findings

| Topic | Result |
|---|---|
| Model choice | Logistic regression, RBF SVM, random forest and gradient boosting span 0.8297 to 0.8352 ROC AUC in repeated CV, a range of 0.0055 against a standard deviation of 0.036. The limit is in the data. |
| Held-out set | On 154 patients the 95% bootstrap interval is about 0.13 wide, so every model's score sits inside every other's. Logistic regression ranked third in CV and first here. |
| Threshold | Four selection rules land between 0.39 and 0.45, away from the 0.5 default. Moving between two of them trades 16 caught cases for 24 fewer false alarms. |
| Missing values | Every patient missing `SkinThickness` or `BloodPressure` is also missing `Insulin`. Imputation strategy makes no measurable difference (0.005 AUC against 0.033 noise). |
| Effect sizes | All eight features have p < 1e-5, but d runs from 1.19 (`Glucose`) down to 0.36 (`BloodPressure`). |
| Clustering | No algorithm beats adjusted Rand 0.16 against the outcome. The k=2 split is stable (ARI 0.83) but tracks age and metabolic load: 17.5% diabetic in one cluster, 55.6% in the other. |
| Explainability | Equally accurate models rely on different features. The forest weights `Age` 12 times more than the linear model, which routes it through `Pregnancies`. Only 4 of 13 coefficient intervals exclude zero. |
| Scaling trap | Clustering raw features scores 3.7x better on silhouette (0.725 vs 0.196) and recovers almost none of the outcome (ARI 0.030 vs 0.161). `Insulin` dominates the distance. |

## Quick start

```bash
conda create -n intro_ds python=3.12
conda activate intro_ds
pip install -e ".[dev]"
```

```bash
python -m pima          # full analysis, headless, ~2 minutes
python -m pima.report   # rebuild docs/index.html
pytest -q               # 65 tests, including a claim audit of the report
jupyter notebook notebooks/
```

Developed on scikit-learn 1.8 and checked on 1.5 with identical results.
`pyproject.toml` caps it below 1.9, which deprecates `SVC(probability=True)`.

## Report

An eight-section write-up with interactive charts. Every number in it is
generated from the processed data, and a test fails if the prose quotes a figure
the data does not produce. Source is `src/pima/report.py`; the build writes
`docs/`, which GitHub Pages serves as static files.

## Notebooks

Read in order. Outputs are committed, so they render on GitHub without running.

| | Notebook | Covers |
|---|---|---|
| 1 | [Data processing](notebooks/01_data_processing.ipynb) | 652 hidden missing values, which columns to unmask, nested missingness |
| 2 | [Exploratory analysis](notebooks/02_exploratory_analysis.ipynb) | Effect sizes vs p-values, why outliers stay, class overlap |
| 3 | [Classification](notebooks/03_classification.ipynb) | Leak-free pipelines, five models, thresholds, one look at the test set |
| 4 | [Clustering](notebooks/04_clustering.ipynb) | Four algorithms, stability, PCA and t-SNE |
| 5 | [Explainability](notebooks/05_explainability.ipynb) | Permutation importance, bootstrap coefficients, partial dependence, surrogate rules, per-patient attribution |

The notebooks are thin: the logic lives in `src/pima/`.

## Project layout

```
src/pima/
  data.py       loading, unmasking disguised zeros, train/test split
  eda.py        distributions, outliers, effect sizes, binned prevalence
  models.py     preprocessing pipelines and the classifier registry
  evaluate.py   cross-validation, thresholds, bootstrap intervals
  cluster.py    clustering sweep, agreement metrics, stability, PCA
  explain.py    permutation importance, coefficients, PDP, surrogate, attribution
  plots.py      notebook figures
  facts.py      every number the report quotes, computed once and cached
  figures.py    Plotly figures for the report
  report.py     builds docs/index.html
tests/          unit tests per module, plus the report's claim audit
data/raw/       diabetes.csv, unmodified
data/processed/ unmasked but never imputed (written by notebook 1)
docs/           the built report
```

## Method

Three rules hold throughout. The tests enforce the first.

1. **Nothing is fitted outside a pipeline.** Imputation and scaling refit on
   every CV fold. `test_preprocessor_is_fit_on_training_data_only` fails if they move out.
2. **Nothing is selected on the test set.** Hyperparameters and thresholds come
   from out-of-fold predictions on the training half. The test set is scored once.
3. **Accuracy is read against a baseline.** 65% of the cohort is non-diabetic,
   so a `majority_class` entry sits in every leaderboard.

| Stage | Approach |
|---|---|
| Cleaning | Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are impossible and become NaN. `Pregnancies` zeros are real and stay; a unit test pins this, since widening the rule would delete 111 patients. The processed file keeps its gaps, because imputing before the split hides leakage. |
| Classification | Five candidates with a majority baseline. Grid search in 5-fold CV, then repeated 5x5 stratified CV. Thresholds from out-of-fold predictions under four objectives (F1, Youden, minimum recall, cost-weighted). Test metrics carry percentile bootstrap intervals. |
| Clustering | k chosen by silhouette, Calinski-Harabasz, Davies-Bouldin and BIC over k=2..10, which disagree. Four algorithms compared on chance-adjusted agreement. Bootstrap stability separates reproducible from meaningful. |
| Explainability | No SHAP. For a linear model on standardised inputs the additive decomposition is exact, and a test asserts it reproduces `predict_proba`. The forest gets partial dependence and a surrogate tree reported with its fidelity (88.3%). |

## Limitations

- **One population.** 768 women of Pima heritage aged 21+. Nothing transfers without revalidation.
- **Small test set.** 154 patients give an interval about 0.13 wide, which limits several conclusions.
- **Insulin is 48.7% missing**, so its coefficient is partly a statement about the imputation.
- **Glucose is close to the diagnostic criterion**, so its dominance is largely definitional.
- **Cross-sectional.** Every association is correlational.
- **Uncalibrated probabilities.** Balanced class weights improve recall and distort probabilities. Treat outputs as a ranking.

## Data and license

`data/raw/diabetes.csv`: 768 rows, 8 features, binary `Outcome`. Originally from
the National Institute of Diabetes and Digestive and Kidney Diseases, redistributed
unmodified. Code is [MIT](LICENSE); the dataset keeps its original terms.

Author: Fahim Ahamed.
