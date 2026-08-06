# Predicting Problematic Internet Use in Children and Adolescents

Can a child's **physical health and sleep** tell us whether their internet use
has become problematic — without asking a single question about their internet
use?

That is the question this project answers, using data from **8,460 children and
adolescents aged 5–22** collected by the Child Mind Institute. The target is the
*Severity Impairment Index* (`sii`), a four-level score:

| `sii` | Meaning |
| :---: | :--- |
| 0 | None |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |

**The catch, and the point of the whole project.** `sii` is calculated directly
from a 20-question parent survey (the PCIAT). A model that can see those 20
answers scores almost perfectly and has learned nothing — it is just reading the
answer off its own input. So every PCIAT column is deleted before any modelling
begins, and the models have to work from body measurements, fitness tests,
bio-impedance readings and a sleep-disturbance questionnaire alone.

That makes the problem genuinely hard, and the honest results below reflect it.

---

## Results at a glance

**Regression track** — predict `sii` as a number, then cut it into the four
classes. Scored with **Quadratic Weighted Kappa** (QWK), which punishes being
three classes wrong far more than being one class wrong.

| Model | Test QWK |
| :--- | ---: |
| **Stacking ensemble** | **0.3712** |
| LightGBM | 0.3613 |
| Random Forest | 0.3247 |
| XGBoost | 0.2946 |
| Ridge | 0.2885 |
| SVR | 0.2801 |
| CatBoost | 0.2714 |
| Lasso | 0.2675 |

**Classification track** — predict the four classes directly. Scored with
**Macro F1**, which weights all four classes equally. That matters: ~69% of
children are class 0, so a model that always guesses "None" would score 69%
accuracy and be useless.

| Model | Accuracy | Macro F1 | Macro AUC |
| :--- | ---: | ---: | ---: |
| XGBoost | 0.63 | **0.38** | 0.7126 |
| CatBoost | 0.62 | 0.37 | 0.7166 |
| Random Forest | 0.58 | 0.37 | **0.7222** |
| LightGBM | 0.59 | 0.36 | 0.7082 |
| PyTorch MLP | 0.58 | 0.35 | 0.6822 |
| SVC | 0.52 | 0.32 | 0.6949 |
| Logistic Regression | 0.50 | 0.32 | 0.6759 |
| Ridge Classifier | 0.51 | 0.31 | 0.6503 |

### What this actually means

**A QWK of 0.37 is moderate agreement.** Far better than chance (0 would be
random), far short of the ~0.80 usually considered good enough to act on
automatically. In practice the model reliably separates "None" from
"Moderate/Severe", but frequently confuses neighbouring levels. It is a
population-level screening aid, not a diagnosis.

**Sleep disturbance is the strongest single predictor.** All four
explainability methods — SHAP, LIME, TREPAN and LORE — independently point at
the SDS sleep score and self-reported daily screen time as the dominant
features. Sleep questionnaires take five minutes and are already routine in
school health checks, so this suggests a cheap two-stage screening protocol:
sleep questionnaire first, full fitness battery only for those who score high.

**The most important result is a failure.** Class 3 ("Severe") has 70 training
examples and 18 test examples, and the ensemble gets **zero** of them right.
Every severely affected child in the test set is missed. In a real deployment
those are precisely the children who most need to be flagged, so any system
built on this model needs an independent safety net for high-risk cases. This
limitation is not a footnote — it is the main finding.

---

## Quick start

```bash
pip install -r requirements.txt
```

On macOS, LightGBM and XGBoost also need the OpenMP runtime:

```bash
brew install libomp
```

Then run any stage of the pipeline:

```bash
python main.py --stage prep-reg
```

The stages, in dependency order:

| Stage | What it does | Needs |
| :--- | :--- | :--- |
| `prep-reg` | Build the regression dataset | the raw CSV |
| `tune-reg` | Tune 8 regression models | `prep-reg` |
| `eval-reg` | Final regression evaluation + stacking | `tune-reg` |
| `prep-clf` | Build the classification dataset | the raw CSV |
| `tune-clf` | Tune 8 classifiers | `prep-clf` |
| `eval-clf` | Final evaluation + explainability + stacking | `tune-clf` |
| `all` | Everything, in order | — |

The two tuning stages are by far the slowest — each runs 8 studies of 20–100
trials, and every trial fits 5 cross-validation folds with a tree-based imputer
inside each one, so budget hours. The preprocessing and evaluation stages sit
in between.

The `optuna trials/` folder starts empty, so **you must run a `tune-` stage
before its matching `eval-` stage**. Skipping it is not a hard error: the
evaluation warns (`WARNING: no trials CSV matching ...` for `eval-reg`,
`File not found: ...` for `eval-clf`) and then falls back to each library's
default hyperparameters. It will still print numbers — just not the ones
reported above. Watch for those lines.

To try tuning out quickly, shorten the studies:

```bash
python main.py --stage tune-reg --models lgbm rf --trials 5
```

Other useful flags: `--no-explain` skips the (slow) SHAP/LIME/TREPAN/LORE step
in `eval-clf`. Running headless? Set a non-interactive plotting backend:

```bash
MPLBACKEND=Agg python main.py --stage eval-reg
```

### Just want to read the code?

Start with `main.py` — it is the whole pipeline in one file, top to bottom.
Each `run_*` function is a stage, and every step it takes is a named call into
`src/`.

Prefer to see the charts? The `notebooks/` folder runs the same pipeline
step by step with plots and explanation. Read them in numerical order.

---

## How the pipeline works

```
dataset/cmi_internet.csv
        │
        ├─ audit               implausible measurements → NaN
        ├─ drop leakage        remove all 20 PCIAT items, CGAS, Season columns
        ├─ split               stratified 80/20 on sii  (random_state=42)
        ├─ filter              drop >50% missing and Spearman |ρ|>0.90 columns
        │                      ── decided on TRAIN rows only ──
        ├─ imputer search      Optuna picks how to fill the ~⅓ missing cells
        ├─ outlier vote        10 detectors; 9/10 agreement flags a row
        │
        └─→ dataset/PostProcessed_{reg,clf}.csv
                    │
                    ├─ tune      Optuna × 5-fold CV → optuna trials/*_trials.csv
                    └─ evaluate  refit on full train, score once on test
```

**Why the order matters.** Everything before the split is a *schema* decision —
it depends on which columns exist, not on any row's values — so it cannot leak
test information. The two filters after the split *are* data-dependent, so they
are measured on the training rows and then applied to the test rows. During
tuning, the scaler, imputer and PCA are re-fitted inside every cross-validation
fold. Getting this wrong is the most common way to produce results that look
good and mean nothing.

### Repository layout

```
tabular/
├── main.py            # single entry point — run the whole pipeline from here
├── requirements.txt
├── src/
│   ├── config.py             # every path, seed and hyperparameter, in one place
│   ├── data_loader.py        # loading, auditing, splitting, imputing, scaling
│   ├── outliers.py           # the 10-detector consensus vote + its diagnostics
│   ├── imputer_search.py     # choosing an imputation strategy
│   ├── models.py             # PyTorch MLP + building models from tuned params
│   ├── tuning.py             # Optuna search spaces and cross-validation
│   ├── metrics.py            # QWK, threshold tuning, reports, plots, stacking
│   └── explainability.py     # SHAP, LIME, TREPAN, LORE
├── notebooks/         # the same pipeline, step by step, with charts
├── dataset/           # raw and processed datasets
├── optuna trials/     # Optuna trial histories, one CSV per model (starts empty)
└── Guidelines/        # the course assignment brief
```

Plots are displayed, not written to disk — run the notebooks to see them.

### Key design decisions

**Imputation.** About a third of all cells are empty. To choose how to fill
them we manufacture ground truth: hide 10% of the values that *are* present,
ask each candidate imputer to reconstruct them, and score the guesses. The two
tracks score differently — regression rewards preserving the *ranking* of
values (an ordinal target cares about order), classification rewards raw
reconstruction accuracy.

**Outliers.** Ten detectors, each built on a different idea of "unusual"
(histogram, distance, density, clustering, angle, depth, isolation). Each flags
its top 1%, and a row is only called an outlier when at least **9 of the 10**
agree. Rows are flagged, never deleted.

**Ordinal thresholds.** The regressors output a continuous number that has to
become one of {0,1,2,3}. Rather than rounding, the three cut points are
optimised to maximise QWK — always fitted on training predictions and then
applied unchanged to test predictions.

**Class imbalance.** Seven resampling strategies were benchmarked (SMOTE,
ADASYN, ENN, Tomek, and combinations). None is used in the final pipeline: the
models use `class_weight='balanced'` instead, which reweights the loss rather
than duplicating rows, and every headline metric is Macro F1 rather than
accuracy.

---

## Known quirks

These are deliberate. They are documented rather than fixed because changing
them would change the published numbers.

- **Imputer selection sees the whole training pool.** The imputer is chosen
  once on all training rows, then re-fitted inside each CV fold. So the
  *choice* has seen rows that later serve as validation data. The bias is small
  (imputation never looks at the target) but the CV scores should be read as
  mildly optimistic.
- **`initial_strategy` falls back to `"mean"`.** The tuned configs in
  `config.py` store this key as `mice_initial_strategy`, but `build_imputer()`
  reads `initial_strategy`, so scikit-learn's `"mean"` default is what actually
  runs.
- **The MLP loses its hyperparameters inside the stacking ensemble.**
  `TorchClassifier.__init__` takes only `**kwargs`, so scikit-learn's
  `get_params()` returns `{}` and `clone()` rebuilds it with defaults. The
  stacked MLP therefore trains with default settings, not tuned ones.
- **Tuning and final evaluation use different imputers on the regression
  track** — `mice` while tuning, `extra_trees` at evaluation.
- **`is_outlier` is a feature during tuning but is dropped at regression
  evaluation.** The two stages genuinely disagree.
- **Several steps are not reproducible run to run.** The Optuna studies are
  unseeded, LODA (one of the ten outlier detectors) has no random seed, and
  LORE draws its neighbourhood from NumPy's global generator. Re-running will
  give close but not identical numbers.
- **`ExtraTreesRegressor(n_jobs=-1)` inside the imputer** sums in a
  nondeterministic order, so imputed values shift by ~1e-14 between runs.

## Data

`dataset/cmi_internet.csv` is the Child Mind Institute "Problematic Internet Use"
dataset. `dataset/data_dictionary.csv` and `dataset/feature_descriptions.md` describe
the columns.

This project covers the **tabular** track only. The accelerometer time-series
analysis lives in the sibling `Timeseries/` project and is not run from here.
