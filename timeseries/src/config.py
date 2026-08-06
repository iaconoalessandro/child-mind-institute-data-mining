"""
STEP 1 — Setup / Configuration.

Every path, random seed, column name and fixed constant used anywhere in the
project lives in this file, so there is exactly one place to look (and one
place to change) when you want to know how the pipeline is wired.

Nothing in here runs a computation; it is pure configuration data.
"""

import os

# ---------------------------------------------------------------------------
# File-system paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the "timeseries" folder: this file lives in timeseries/src/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "dataset_dir": os.path.join(PROJECT_ROOT, "dataset"),
    "figures_dir": os.path.join(PROJECT_ROOT, "figures"),
    # A pickled Python list of 4,437 DataFrames, one per subject. Despite what
    # the dataset instructions say, the file shipped here is NOT gzipped, so it
    # is opened with a plain open() rather than gzip.open().
    "raw_data": os.path.join(PROJECT_ROOT, "dataset",
                             "raw_actigraphy_dataset.pkl"),
    # The four arrays written by the preparation stage. Every later stage reads
    # these instead of touching the raw pickle again.
    "X_train": os.path.join(PROJECT_ROOT, "dataset", "train_features.npy"),
    "X_test": os.path.join(PROJECT_ROOT, "dataset", "test_features.npy"),
    "y_train": os.path.join(PROJECT_ROOT, "dataset", "train_labels.npy"),
    "y_test": os.path.join(PROJECT_ROOT, "dataset", "test_labels.npy"),
    # One row per (subject, feature, timestamp) value the Hampel filter removed.
    "outlier_log": os.path.join(PROJECT_ROOT, "dataset",
                                "detected_outliers.csv"),
    # The vendored third-party PULSAR feature extractor (GPL-3.0, ICDM 2025).
    "pulsar_dir": os.path.join(PROJECT_ROOT, "PULSAR"),
}

# ---------------------------------------------------------------------------
# Random seeds
# ---------------------------------------------------------------------------
# Almost every seed in the project is 42. They are listed separately (rather
# than as a single constant) so each usage site reads self-documenting at the
# call site, e.g. SEEDS["train_test_split"] rather than a bare 42.
SEEDS = {
    "train_test_split": 42,     # train_test_split of the subject index
    "kmeans": 42,               # KMeans and TimeSeriesKMeans
    "pca": 42,                  # PCA(random_state=...)
    "tsne": 42,                 # t-SNE on the DTW distance matrix
    "shapelet_order": 42,       # order in which training subjects are scanned
    "random_shapelet": 100,     # the K=20 random-shapelet baseline
    "jitter": 42,               # Gaussian jitter added for aeon classifiers
    "extra_trees": 42,          # ExtraTreesClassifier behind PULSAR
    "lore": 42,                 # LORE neighbourhood sampling + surrogate tree
}

# ---------------------------------------------------------------------------
# STEP 2 — Data preparation constants
# ---------------------------------------------------------------------------
# Dropped from every subject's DataFrame before modelling. `sii_binary` is the
# target, and `id` is a static per-subject tracker that a model could memorise;
# the other three are calendar metadata, not sensor signal.
COLUMNS_TO_DROP = ["weekday", "quarter", "relative_date_PCIAT", "sii_binary",
                   "id"]

# The eight columns that survive the drop, in the exact order pandas leaves
# them. This order defines the last axis of the saved .npy arrays, so the
# channel indices below depend on it.
FEATURE_COLUMNS = ["X", "Y", "Z", "enmo", "anglez", "non-wear_flag", "light",
                   "battery_voltage"]

# Columns that get Z-score normalised. `non-wear_flag` is deliberately absent:
# it is a 0/1 indicator, and standardising it would destroy that meaning.
SCALED_FEATURES = ["X", "Y", "Z", "enmo", "anglez", "light", "battery_voltage"]

# Columns the Hampel outlier filter skips. A rolling median over a binary flag
# is meaningless, and `id` is a constant tracker rather than a signal.
HAMPEL_SKIP_COLUMNS = {"id", "non-wear_flag"}

# Per-feature Hampel filter settings: `window` is the rolling-median width in
# time steps and `sigma` is how many scaled MADs a point must sit away from
# that median before it counts as an outlier. Noisier or spikier channels get
# a looser sigma; the slow-moving battery trace gets a wide window instead.
HAMPEL_CONFIGS = {
    "X":               {"window": 5,  "sigma": 4.5},
    "Y":               {"window": 5,  "sigma": 4.5},
    "Z":               {"window": 5,  "sigma": 4.5},
    "enmo":            {"window": 5,  "sigma": 5.5},
    "anglez":          {"window": 5,  "sigma": 4.5},
    "light":           {"window": 7,  "sigma": 5.0},
    "battery_voltage": {"window": 15, "sigma": 3.0},
}

# Fallback for any column not named in HAMPEL_CONFIGS.
HAMPEL_DEFAULT = {"window": 7, "sigma": 4.5}

TEST_SIZE = 0.20  # hold-out fraction of the stratified subject split

# ---------------------------------------------------------------------------
# Dataset shape
# ---------------------------------------------------------------------------
N_TIMESTEPS = 200        # time steps per subject
MINUTES_PER_STEP = 30    # each step is a 30-minute average

# The two channels every downstream model uses, given as indices into
# FEATURE_COLUMNS: 3 = enmo (movement intensity), 4 = anglez (wrist posture).
# The raw X/Y/Z axes are excluded because of naming inconsistencies in the
# source dataset, and light/battery carry no behavioural signal.
MODEL_CHANNELS = [3, 4]
MODEL_CHANNEL_NAMES = ["enmo", "anglez"]

# Human-readable names for the binary target.
CLASS_NAMES = ["Healthy", "Unhealthy"]
CLASS_NAMES_LONG = ["Non-Problematic (Class 0)", "Problematic (Class 1)"]

# ---------------------------------------------------------------------------
# STEP 3 — Approximation, clustering and matrix profile
# ---------------------------------------------------------------------------
PAA_N_SEGMENTS = 40      # 200 time steps compressed to 40 PAA segments
N_CLUSTERS = 7           # clusters used by both clustering algorithms
K_SEARCH_RANGE = range(2, 11)   # k values scored by the elbow/silhouette sweep

# Sakoe-Chiba band for DTW: a radius of 4 over a 40-segment sequence limits the
# warping corridor to ~10% of the series length, which is what makes the
# pairwise DTW computation tractable.
SAKOE_CHIBA_RADIUS = 4
DTW_N_JOBS = 4

TSNE_PERPLEXITY = 15
TSNE_MAX_ITER = 1000

MATRIX_PROFILE_WINDOW = 30   # subsequence length m for stumpy.stump
MATRIX_PROFILE_MAX_MOTIFS = 10
MATRIX_PROFILE_TOP_DISCORDS = 5

# ---------------------------------------------------------------------------
# STEP 4 — Shapelet analysis
# ---------------------------------------------------------------------------
# The 15 hand-picked candidate shapelets, as
# (id, subject_index, channel, start_index, length). Channel 0 is enmo and
# channel 1 is anglez, matching MODEL_CHANNELS above. They deliberately span
# three lengths (20/30/40) and both channels so the leaderboard can show which
# combination discriminates best.
SHAPELET_CANDIDATES = [
    (0, 10, 0, 30, 30),
    (1, 10, 1, 100, 30),
    (2, 25, 0, 50, 20),
    (3, 25, 1, 120, 20),
    (4, 40, 0, 80, 40),
    (5, 40, 1, 40, 40),
    (6, 55, 0, 100, 30),
    (7, 55, 1, 60, 30),
    (8, 70, 0, 20, 20),
    (9, 70, 1, 140, 20),
    (10, 85, 0, 110, 40),
    (11, 85, 1, 10, 40),
    (12, 100, 0, 60, 30),
    (13, 115, 0, 40, 30),
    (14, 130, 1, 80, 30),
]

# Entropy pruning only starts after this fraction of the training set has been
# scanned, then re-checks every PRUNING_CHECK_FRACTION of it. The guard window
# stops a candidate being discarded on the strength of a handful of samples.
PRUNING_GUARD_FRACTION = 0.20
PRUNING_CHECK_FRACTION = 0.10

# Divisor guard that keeps Z-normalisation finite on flat (zero-variance)
# windows. Present in every shapelet distance function.
ZNORM_EPSILON = 1e-8

# The random-shapelet baseline: K subsequences drawn uniformly at random.
RANDOM_SHAPELET_COUNT = 20
RANDOM_SHAPELET_LENGTHS = [20, 30, 40]

# ---------------------------------------------------------------------------
# STEP 5 — Classification
# ---------------------------------------------------------------------------
CV_FOLDS = 4    # folds used for every out-of-fold prediction and CV score

# Decision thresholds swept when converting predicted probabilities into hard
# labels. 0.1 to 0.9 in steps of 0.01.
THRESHOLD_GRID = (0.1, 0.9, 81)

# Optuna budget for the only tuned model. The other five classifiers are run at
# their published defaults because tuning them is computationally prohibitive.
KNN_OPTUNA_TRIALS = 10
KNN_N_NEIGHBORS_RANGE = (1, 20)

# Standard deviation of the Gaussian jitter added before the aeon classifiers
# run. RDSTClassifier raises a fatal ValueError on a completely flat channel;
# 1e-6 is far below any meaningful signal but satisfies its variance check.
JITTER_SIGMA = 1e-6

# Sakoe-Chiba band for sktime's DTW k-NN, expressed as a fraction of the series
# length. Without it, unbounded DTW over 3,549 multivariate subjects never
# finishes.
DTW_WINDOW_FRACTION = 0.1

# MiniRocket / MultiRocket-Hydra convolution kernels.
MINIROCKET_N_KERNELS = 10000
# The ROCKET feature space is high-dimensional, so the internal logistic
# regression needs far more iterations than its default 1000 to converge.
MINIROCKET_MAX_ITER = 5000

RDST_MAX_SHAPELETS = 1000

# PULSAR feature-extractor settings, applied identically to each channel.
PULSAR_PARAMS = {
    "time_series_representations": ["original", "periodogram", "derivative",
                                    "autoregressive"],
    "local_statistics": ["mean", "stdev", "slope", "min", "max", "iqr",
                         "median"],
    "global_statistics": ["max_pooling", "mean_pooling", "min_pooling",
                          "median_pooling", "iqr_pooling", "stdev_pooling"],
    "list_interval_lengths": [7, 9, 11],
    "depth_local_features": 3,
    "percentage_top_local_features": 40,
    "max_dilation": 16,
}

# The classifier trained on PULSAR's concatenated features.
PULSAR_ET_PARAMS = {
    "n_estimators": 50,
    "criterion": "entropy",
    "max_features": 0.10,
    "random_state": SEEDS["extra_trees"],
    "class_weight": "balanced",
}

# LORE (Local Rule-based Explanations) settings: how many synthetic neighbours
# to draw around the instance being explained, how wide to spread them (as a
# fraction of each feature's standard deviation), and how deep the surrogate
# decision tree may grow.
LORE_N_NEIGHBOURS = 800
LORE_NOISE_SCALE = 0.1
LORE_EPSILON = 1e-5
LORE_TREE_MAX_DEPTH = 4
LORE_EXPLAIN_INDICES = [0, 5, 10]

# ---------------------------------------------------------------------------
# STEP 6 — Sequential pattern mining
# ---------------------------------------------------------------------------
SPM_CHANNEL = 3          # enmo, as an index into FEATURE_COLUMNS
SPM_PAA_WINDOW = 10      # 200 time steps averaged into 20 blocks of 10
# Percentile cut points that turn a numeric PAA value into one of L / M / H.
# Chosen so the three symbols are equally frequent across the whole dataset.
SPM_QUANTILES = [33.33, 66.67]
SPM_SYMBOLS = ["L", "M", "H"]

# Minimum support for a pattern to count as frequent, as a fraction of the
# cohort. 20% is low by Apriori standards, but the max_gap=1 constraint makes
# matching much stricter, so a higher bar returns almost nothing.
SPM_MIN_SUPPORT = 0.20
# max_gap=1 means pattern elements must be strictly contiguous in time.
SPM_MAX_GAP = 1
SPM_TOP_N = 10           # how many patterns to print and plot per cohort
