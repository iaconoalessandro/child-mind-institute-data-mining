# Predicting Problematic Internet Use in Children and Adolescents

Two independent attempts at the same question, using data from the Child Mind Institute:
can problematic internet use be detected without asking about internet use at all —
once from physical and sleep measurements, once from a week of wrist accelerometer
recordings.

Academic project — Data Mining 2, MSc in Data Science and Business Informatics,
University of Pisa, a.a. 2025/2026. Coursework, done in a pair with Irene Mungo.

## The two tracks

| | [`tabular/`](./tabular) | [`timeseries/`](./timeseries) |
| :--- | :--- | :--- |
| Input | body measurements, fitness tests, bio-impedance, sleep questionnaire | wrist accelerometer, 200 steps of 30-minute averages |
| Subjects | 8,460 | 4,437 |
| Target | `sii`, 4 ordinal levels | binary, problematic or not |
| Headline | QWK 0.3712 (stacking) | Macro F1 0.6259 (MultiRocket-Hydra) |

Each track has its own README with the full method, results and caveats. This page is
the summary; the detail is there.

## Why the problem is harder than it looks

The target `sii` is computed directly from a 20-question parent survey about the child's
internet use. Any model allowed to see those 20 answers scores almost perfectly while
learning nothing — it is reading its answer off its own input. Every PCIAT column is
therefore deleted before modelling, and the models work from physical, sleep and
movement data alone. The modest numbers below are what that constraint costs, and the
constraint is the point of the exercise.

## Data

**Tabular** (`tabular/dataset/cmi_internet.csv`) — 8,460 children and adolescents aged
5 to 22, 82 columns. The target is badly imbalanced: 5,833 at level 0 (None), 1,587 at 1
(Mild), 952 at 2 (Moderate), and **88 at level 3 (Severe)** across the whole dataset.
About a third of all cells are empty, and some recorded measurements are physically
implausible and are audited to NaN before anything else.

**Time series** (`timeseries/dataset/raw_actigraphy_dataset.pkl`) — 4,437 subjects, each
about 4.2 days of wrist sensor data stored as 200 time steps across 8 channels
(tri-axial acceleration, ENMO, wrist angle, light, battery, non-wear flag). Split by
subject into 3,549 training and 888 test. Class balance 66.6% / 33.4%.

## Approach

**Tabular.** Audit implausible values, drop the leaking PCIAT columns, stratified 80/20
split, then column filtering decided on training rows only. An Optuna search picks the
imputation strategy by hiding 10% of known values and scoring the reconstruction. Ten
outlier detectors vote, and a row is flagged only when 9 of 10 agree — flagged, never
deleted. Then two parallel tracks: regression with tuned ordinal cut points, and direct
4-class classification, each over 8 model families tuned with Optuna under 5-fold CV,
followed by a stacking ensemble and SHAP, LIME, TREPAN and LORE explanations.

**Time series.** Subject-level split first, then a per-channel Hampel filter and
interpolation on training rows only, then per-channel z-scoring. Four independent
analyses on the prepared arrays: DTW clustering with matrix profile, shapelet mining,
six classifiers with thresholds tuned on out-of-fold predictions, and sequential pattern
mining over discretised activity levels.

Only `enmo` (movement intensity) and `anglez` (wrist posture) are modelled. The raw X/Y/Z
axes are excluded because of naming inconsistencies in the source data, and light and
battery voltage carry no behavioural signal.

## What did not work

**The severe cases are missed entirely.** Level 3 has 88 subjects in the whole dataset,
18 of them in the test set, and the model gets zero of them right — precision, recall and
F1 all 0.00 for that class. In any real deployment those are exactly the children the
system exists to find. This is the single most important result in the project and it is
a failure.

**No local pattern separates the classes.** The shapelet search evaluated 15 carefully
chosen candidates; the best scored an information gain of 0.0032 bits out of a possible
1.0, and 20 randomly drawn subsequences did slightly better on average. Seven of the 15
were pruned before they finished. This is why the classifiers that work at all are the
ones combining thousands of weak features rather than relying on one strong one.

**There are no natural clusters.** Silhouette peaks at 0.106 and never leaves the
0.06–0.11 band across every k from 2 to 10. Children's daily routines form a continuum,
not distinct behavioural types. The seven clusters used later are a descriptive partition
for the matrix-profile analysis, not a claim about real structure.

**Resampling was benchmarked and then discarded.** Seven strategies — SMOTE, ADASYN, ENN,
Tomek and combinations — were compared and none is used in the final pipeline.
`class_weight='balanced'` reweights the loss instead of duplicating rows, and every
headline metric is Macro F1 rather than accuracy.

**Beating the majority-class baseline on accuracy is not evidence of anything here.**
69% of the tabular subjects are class 0 and 66.7% of the time-series subjects are class
0. Two time-series models exceed that accuracy mainly by predicting the majority class
more often, not by finding at-risk children better. Accuracy is reported for context only.

Both sub-projects also document their own known quirks — leakage subtleties, unseeded
components, and places where two pipeline stages genuinely disagree. Those lists are in
the respective READMEs and are worth reading before trusting any single number.

## Results

**Tabular, regression track** — Quadratic Weighted Kappa, which penalises being three
classes wrong far more than one class wrong.

| Model | Test QWK |
| :--- | ---: |
| Stacking ensemble | 0.3712 |
| LightGBM | 0.3613 |
| Random Forest | 0.3247 |
| XGBoost | 0.2946 |

**Tabular, classification track** — Macro F1 across the four classes.

| Model | Accuracy | Macro F1 | Macro AUC |
| :--- | ---: | ---: | ---: |
| XGBoost | 0.63 | 0.38 | 0.7126 |
| CatBoost | 0.62 | 0.37 | 0.7166 |
| Random Forest | 0.58 | 0.37 | 0.7222 |

**Time series** — six classifiers on the same held-out 888 subjects.

| Model | Accuracy | Macro F1 | AUC-ROC |
| :--- | ---: | ---: | ---: |
| MultiRocket-Hydra | 0.6847 | 0.6259 | 0.6216 |
| k-NN (DTW) | 0.6441 | 0.6049 | 0.6064 |
| PULSAR (early fusion) | 0.7016 | 0.5978 | 0.7295 |
| Naive baseline (always "non-problematic") | 0.6667 | 0.400 | 0.500 |

### What these numbers mean

A QWK of 0.37 is moderate agreement — clearly better than chance, well short of the ~0.80
usually needed before anything is acted on automatically. The model separates "None" from
"Moderate or Severe" reliably and confuses neighbouring levels often. On the time-series
side, PULSAR's AUC of 0.73 shows wrist actigraphy does carry signal, but every model sits
in a narrow 0.58–0.63 Macro F1 band. Both tracks produce population-level screening
correlates, not individual diagnoses.

Two findings are consistent across every method tried. Sleep disturbance and self-reported
screen time dominate the tabular feature importances under all four explainability
methods. And reduced physical activity is the mechanism on the sensor side: problematic
children spend 38.4% of their time in the lowest activity tercile against 30.8% for the
rest, and 26.4% of them show 30 continuous hours of low activity — a pattern that does not
clear the support threshold in the non-problematic group at all.

## How to run

The two tracks are independent and do not share code or environments. Install and run
each from inside its own folder.

```bash
git clone https://github.com/iaconoalessandro/child-mind-institute-data-mining.git
cd child-mind-institute-data-mining
```

```bash
cd tabular
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --stage all
```

```bash
cd timeseries
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --stage all
```

On macOS the gradient-boosting libraries also need the OpenMP runtime:
`brew install libomp`.

Both entry points take `--stage` and run a single step at a time; `tabular` needs its
`tune-` stages run before the matching `eval-` stages, and `timeseries` needs `prep`
before anything else. Budget hours for the tuning and classification stages. Each
README lists the stages, their dependencies and the flags for cutting the work down.

To read rather than run, start with `main.py` in either folder — the whole pipeline is
one file, top to bottom. The `notebooks/` folders run the same steps with plots.

## Tech stack

Taken from the two requirements files.

| Purpose | Library |
| :--- | :--- |
| Data handling | numpy, pandas, scipy |
| Models and CV | scikit-learn |
| Gradient boosting | LightGBM, XGBoost, CatBoost |
| Neural network | PyTorch |
| Hyperparameter search | Optuna |
| Time series | sktime, tslearn, aeon, stumpy, statsmodels, numba |
| Outlier detection and projections | pyod, umap-learn, pacmap |
| Class imbalance | imbalanced-learn |
| Explainability | SHAP, LIME, plus TREPAN and LORE implemented in `src/` |
| Plots | matplotlib, seaborn |

## Repository layout

```
tabular/                    physical and sleep measurements -> sii (4 classes)
timeseries/                 wrist accelerometer -> binary problematic use
Project_DM2_Iacono_Mungo.pdf   the submitted report
Project Guidelines.pdf         the course assignment brief
```

`timeseries/PULSAR/` is a vendored, unmodified copy of the reference implementation from
the ICDM 2025 PULSAR paper, used under its GPL-3.0 licence and trimmed to the modules
this project imports. The original is at <https://github.com/stevcabello/PULSAR>.

## Authors

Alessandro Iacono and Irene Mungo.
