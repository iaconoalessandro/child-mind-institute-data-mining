# Reading Problematic Internet Use from a Wrist Sensor

Can a week of **wrist movement data** tell us whether a child's internet use has
become problematic — without asking them a single question?

That is what this project tests, using accelerometer recordings from **4,437
children and adolescents** collected by the Child Mind Institute. Each child
wore a wrist sensor for about 4.2 days; the recording is stored as 200 time
steps of 30-minute averages across several sensor channels.

The target is binary:

| `sii_binary` | Meaning |
| :---: | :--- |
| 0 | Non-problematic internet use (66.6% of children) |
| 1 | Problematic — originally mild, moderate or severe (33.4%) |

No questionnaire answers are used. The models see movement, posture and light
alone, which makes the problem genuinely hard — and the honest results below
reflect that.

---

## Results at a glance

Six classifiers, all evaluated once on the same held-out 20% of subjects.
**Macro F1** is the headline number: it weights both classes equally, so a
model that ignores the at-risk minority is properly penalised. Accuracy is
shown for context only.

| Model | Accuracy | Macro F1 | AUC-ROC |
| :--- | ---: | ---: | ---: |
| MultiRocket-Hydra | 0.6847 | **0.6259** | 0.6216 |
| k-NN (DTW) | 0.6441 | 0.6049 | 0.6064 |
| RDST | **0.7140** | 0.6000 | 0.6030 |
| PULSAR (early fusion) | 0.7016 | 0.5978 | **0.7295** |
| k-NN (Euclidean) | 0.6318 | 0.5780 | 0.6256 |
| MiniRocket | 0.6115 | 0.5772 | 0.6023 |
| *Naive baseline (always guess "non-problematic")* | *0.6667* | *0.400* | *0.500* |

> **These numbers move slightly between runs.** RDST, MiniRocket and
> MultiRocket-Hydra all leave `random_state` unset, so their randomly sampled
> kernels and shapelets differ each time; PULSAR's feature selection is
> unseeded too. Re-running typically shifts Macro F1 by ~0.01. The two k-NN
> models and everything in notebooks 01, 03 and 05 are fully deterministic and
> reproduce exactly. See *Known quirks* below.

### What this actually means

**The signal is real but weak.** PULSAR's AUC of 0.73 is clearly above chance,
so wrist actigraphy does carry information about problematic internet use. But
every model sits in a narrow 0.58–0.63 Macro F1 band, and the two models that
beat the naive accuracy baseline do so mostly by predicting the majority class
more often — not by finding at-risk children better. This is a population-level
correlate, not an individual screening tool.

**Reduced physical activity is the mechanism.** Every stage of the analysis
points the same way. Problematic children spend more time in the lowest
activity tercile (38.4% of their blocks vs 30.8%) and less in the highest
(26.7% vs 36.6%). The pattern mining makes it concrete: **26.4% of problematic
children show 30 continuous hours of low activity** (`L→L→L→L→L→L`), a pattern
that does not clear the support bar in the non-problematic group at all.

**No single local pattern separates the classes.** The shapelet search
(notebook 03) is the clearest negative result: the best of fifteen carefully
chosen candidates scores an information gain of **0.0032 bits out of a possible
1.0**, and twenty randomly drawn subsequences do slightly *better* on average.
That is why the models that work at all are the ones combining thousands of
weak features rather than relying on one strong one.

**The clusters are not real clusters.** Silhouette scores peak around 0.10
across every k from 2 to 10 — children's daily routines form a continuum, not
distinct behavioural types. The seven clusters used in notebook 02 are a
descriptive partition for the matrix-profile analysis, not a claim about
natural structure.

---

## Quick start

```bash
pip install -r requirements.txt
```

Then run any stage of the pipeline:

```bash
python main.py --stage prep
```

The stages:

| Stage | What it does | Needs | Roughly |
| :--- | :--- | :--- | :--- |
| `prep` | Build the four `.npy` arrays from the raw pickle | the pickle | ~20 min |
| `cluster` | PAA, DTW clustering, matrix profile | `prep` | ~1 h |
| `shapelets` | Shapelet search and rule extraction | `prep` | ~10 min |
| `classify` | Train and evaluate the six classifiers | `prep` | hours |
| `patterns` | Sequential pattern mining | `prep` | ~1 min |
| `all` | Everything, in order | — | — |

**`prep` is the only hard dependency** — it writes the arrays every other stage
loads. The other four are independent and can run in any order. If you skip it,
the others fail immediately with a clear message rather than silently producing
wrong numbers.

`classify` is the slow one: DTW k-NN and the two ROCKET variants each
cross-validate over 3,549 subjects. To try a subset:

```bash
python main.py --stage classify --models knn-euclidean minirocket
```

Other useful flags: `--no-explain` skips the (slow) PULSAR feature-importance
and LORE step, and `--no-plots` skips every figure. Running headless? Set a
non-interactive plotting backend:

```bash
MPLBACKEND=Agg python main.py --stage cluster --no-plots
```

### Just want to read the code?

Start with [`main.py`](main.py) — it is the whole pipeline in one file, top to
bottom. Each `run_*` function is a stage, and every step it takes is a named
call into `src/`.

Prefer to see the charts? The [`notebooks/`](notebooks/) folder runs the same
pipeline step by step with plots and explanation. Read them in numerical order;
each one names the `main.py` command it is equivalent to.

---

## How the pipeline works

```
dataset/raw_actigraphy_dataset.pkl   (4,437 subjects x 200 steps)
        │
        ├─ split            stratified 80/20 by subject  (random_state=42)
        ├─ Hampel filter    per-channel spike removal  ── TRAIN ROWS ONLY ──
        ├─ interpolate      linear + ffill/bfill fills the holes it left
        ├─ normalise        Z-score per channel, statistics from TRAIN only
        │
        └─→ dataset/{train,test}_features.npy   (subjects, 200, 8)
            dataset/{train,test}_labels.npy
                    │
                    ├─ cluster     PAA → DTW k-means / hierarchical → matrix profile
                    ├─ shapelets   15 candidates → information gain → rule
                    ├─ classify    6 models, threshold tuned on out-of-fold CV
                    └─ patterns    L/M/H symbols → GSP → contrast patterns
```

**Why the order matters.** The split happens at *subject* level and comes
first, before anything data-dependent. All 200 rows of a child move together,
so no part of a test subject is ever seen during training. The scalers are
fitted on training rows only. Inside the classifiers, every `StandardScaler`
lives in a pipeline so it is refitted within each cross-validation fold, and
every decision threshold is tuned on out-of-fold predictions rather than on the
test set. Getting this wrong is the most common way to produce results that
look good and mean nothing.

### Repository layout

```
timeseries/
├── main.py                  # single entry point — run the whole pipeline from here
├── requirements.txt
├── src/
│   ├── config.py            # every path, seed and constant, in one place
│   ├── data_loader.py       # loading, splitting, outlier filtering, normalising
│   ├── plots.py             # the three preparation-stage diagnostic figures
│   ├── clustering.py        # PAA, DTW clustering, projections, matrix profile
│   ├── shapelets.py         # subsequence distance, information gain, pruning
│   ├── models.py            # the six classifiers
│   ├── metrics.py           # evaluation, threshold search, comparison table
│   ├── explainability.py    # PULSAR feature importances + LORE
│   ├── sequential_patterns.py  # discretisation + GSP + contrast mining
│   └── utils.py             # shared helpers
├── notebooks/               # the same pipeline, step by step, with charts
├── dataset/      # the raw pickle and the prepared .npy arrays
├── PULSAR/                  # vendored third-party feature extractor (GPL-3.0)
├── figures/                 # figures written by the pipeline
└── CMI_TimeSeries_Report.md # the written project report
```

### Key design decisions

**Only two channels are modelled.** `enmo` (movement intensity) and `anglez`
(wrist posture). The raw X/Y/Z accelerometer axes are excluded because of
naming inconsistencies in the source dataset; `light` and `battery_voltage`
carry no behavioural signal. The prepared arrays still hold all eight columns,
so this choice is one line in `config.MODEL_CHANNELS` rather than a
preprocessing decision baked into the data.

**Outliers are filtered, not deleted.** A per-channel Hampel filter blanks out
sensor spikes (about 175,000 individual values across the training set) and
interpolation fills the holes. No subject is ever dropped, so the class balance
is preserved exactly. Every removed value is logged to
`dataset/detected_outliers.csv` so the filter's effect is
auditable.

**Z-score, not min-max.** The classifiers all measure *distance* between
points. Z-scoring strips out the baseline offset that differs between devices
and bodies while preserving the shape and relative size of activity peaks — the
part that carries behavioural meaning.

**Thresholds are tuned, not assumed.** With a 2:1 class imbalance, the default
0.5 probability cut-off badly under-predicts the minority class. Four of the
six models sweep the threshold from 0.1 to 0.9 and keep whichever maximises
Macro F1 — always measured on out-of-fold cross-validation predictions.

**Three library-level fixes**, each documented at its call site:
a Sakoe-Chiba band (10% of series length) so DTW terminates at all; Gaussian
jitter of σ = 1e-6 so `RDSTClassifier` tolerates flat channels instead of
aborting; and `max_iter=5000` so MiniRocket's internal logistic regression
converges in the high-dimensional ROCKET feature space.

---

## Known quirks

These are deliberate. They are documented rather than fixed because changing
them would change the published numbers.

- **The test set is never outlier-filtered or imputed.** Only training subjects
  go through the Hampel filter. This avoids any risk of leakage, but it does
  mean the two halves are not preprocessed identically — the test subjects keep
  their raw spikes.
- **DTW clustering treats the data as univariate.** `build_dtw_input` hands DTW
  the *flattened* PAA matrix: the eight channels stay concatenated end to end
  and are treated as a single 320-step series per subject, rather than as 40
  steps across 8 channels. Every clustering result and centroid was produced
  this way.
- **Four of the six classifiers are not reproducible run to run.** RDST,
  MiniRocket and MultiRocket-Hydra all accept a `random_state` that is left
  unset, so the kernels and shapelets they sample differ every run — two
  identical MiniRocket fits in the same process disagree on about 9% of their
  predictions. PULSAR's feature selection calls Python's unseeded
  `random.sample` (`PULSAR/utils.py`). The Optuna study for k-NN (Euclidean) is
  also unseeded, so its chosen `n_neighbors` can vary, though the search space
  is small enough that it reliably lands on 12.

  Setting `random_state` on those three classifiers would fix this, but it
  would also change the published numbers, so it is left as the original
  analysis ran it.
- **PULSAR's final calibration is optimistic.** The forest is fitted on the
  training set, then wrapped in `CalibratedClassifierCV(cv="prefit")` and
  calibrated on that same training set. The cross-validated threshold search
  above it is honest, but the probabilities themselves are fitted on data the
  model has already seen.
- **The shapelet speedup audit uses an analytic baseline.** The
  "brute-force operations" figure is computed as
  `n_subjects x (n_timesteps - L + 1) x L` rather than measured by running a
  second, slower pass. The early-abandon count is measured directly.
- **Sequential pattern durations assume contiguous recording.** Each symbol
  covers 10 time steps of 30 minutes = 5 hours, so a 6-symbol pattern is
  reported as 30 hours. That treats the 200 steps as one unbroken stretch,
  which is what the dataset represents.
- **Results are sensitive to library versions.** `sktime`, `tslearn` and `aeon`
  all change defaults between minor releases. The numbers above were produced
  with sktime 0.40.1, tslearn 0.6.3, aeon 1.4.0, scikit-learn 1.7.2 and
  numpy 2.3.5.

### What changed in the refactor

Behaviour is unchanged — `patterns` and `shapelets` reproduce their previous
console output line for line, and `prep` reproduces its arrays bit for bit. Two
pieces of genuinely dead code were removed along the way:

- A standalone `StandardScaler` fitted in the classification notebook whose
  output was never used (each model scales inside its own pipeline).
- A re-merge of `non-wear_flag` after normalisation, which was a no-op because
  the normalisation step already copies the column through untouched.

---

## Data

`dataset/raw_actigraphy_dataset.pkl` holds a pickled Python list of
4,437 DataFrames, one per subject, extracted from the Child Mind Institute
"Problematic Internet Use" dataset. Rows are time steps, columns are sensor
channels plus `id` and `sii_binary`.

Note that despite what `dataset/dataset_instructions.txt` says, the file
shipped here is **not** gzipped — it is opened with a plain `open()`.

| Channel | Physical meaning |
| :--- | :--- |
| X, Y, Z | Raw tri-axial accelerometer readings (wrist orientation) |
| ENMO | Euclidean Norm Minus One — overall movement intensity, in *g* |
| ANGLEZ | Wrist angle from horizontal — posture and sleep/wake proxy |
| LIGHT | Ambient light — indoor/outdoor and time-of-day context |
| BATTERY_VOLTAGE | Device health; not used analytically |
| NON-WEAR_FLAG | Whether the device was actually being worn |

`PULSAR/` is a vendored copy of the reference implementation from the ICDM 2025
paper *"PULSAR: Advancing Interval-Based Time Series Classification to
State-of-the-Art Performance"*, used under its GPL-3.0 licence. **No source
file has been modified**, but the copy is trimmed to the six modules this
project imports plus the licence and the authors' README — the UCR benchmark
datasets, resample indices, published result tables and the authors' own
`main.ipynb` were removed, since nothing here reads them. The full original is
at <https://github.com/stevcabello/PULSAR>.

This project covers the **time-series** track only. The tabular analysis of the
same study lives in the sibling `tabular/` project and is not run from here.
