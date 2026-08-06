"""
STEP 1 — Setup / Configuration.

Every path, random seed and fixed hyperparameter used anywhere in the project
lives in this file, so there is exactly one place to look (and one place to
change) when you want to know how the pipeline is wired.

Nothing in here runs a computation; it is pure configuration data.
"""

import os

# ---------------------------------------------------------------------------
# File-system paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the "tabular" folder: this file lives in tabular/src/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    # Optuna writes one trials CSV per model here, and the evaluation stages
    # read them back to rebuild the tuned models.
    "results_dir": os.path.join(PROJECT_ROOT, "optuna trials"),
    # Raw survey/fitness export from the Child Mind Institute study.
    "raw_data": os.path.join(PROJECT_ROOT, "dataset", "cmi_internet.csv"),
    # Cleaned + split datasets written by the two preprocessing stages.
    "reg_processed": os.path.join(PROJECT_ROOT, "dataset", "PostProcessed_reg.csv"),
    "clf_processed": os.path.join(PROJECT_ROOT, "dataset", "PostProcessed_clf.csv"),
}

# ---------------------------------------------------------------------------
# Random seeds
# ---------------------------------------------------------------------------
# Every seed in the project is 42. They are listed separately (rather than as a
# single constant) so that each usage site reads self-documenting at the call
# site, e.g. SEEDS["cv_shuffle"] rather than a bare 42.
SEEDS = {
    "random_state": 42,      # scikit-learn estimators, Optuna, general use
    "catboost_seed": 42,     # CatBoost calls the argument random_seed
    "train_test_split": 42,  # train_test_split
    "cv_shuffle": 42,        # StratifiedKFold(shuffle=True)
    "pca": 42,               # PCA(random_state=...)
    "surrogate_tree": 42,    # TREPAN / LORE surrogate decision trees
    "mask_rng": 42,          # create_mask() random generator
}

# ---------------------------------------------------------------------------
# Imputation settings
# ---------------------------------------------------------------------------
# These dictionaries are the *winning* configurations from the Optuna imputer
# studies in the preprocessing stages. They are passed to
# data_loader.build_imputer(), which reads the keys it recognises.
#
# IMPORTANT, and easy to miss: build_imputer() reads the key "initial_strategy".
# None of the three dictionaries below define that key (they define
# "mice_initial_strategy" instead), so all three fall back to the default
# "mean" initial strategy. This is the behaviour that produced the reported
# results, so it is kept exactly as-is. Do not rename the keys unless you also
# intend to re-run every downstream number.

# Used while TUNING the regression models.
IMPUTER_PARAMS_REG = {
    "imputer_choice": "mice",
    "mice_max_iter": 10,
    "mice_initial_strategy": "median",
    "mice_et_estimators": 20,
    "mice_et_max_depth": 4,
}

# Used while EVALUATING the regression models. This is deliberately different
# from IMPUTER_PARAMS_REG: it is the top-scoring imputer from the regression
# Optuna study (extra_trees, CV score 0.3730), whereas tuning ran on 'mice',
# which converged faster and gave more stable cross-validation residuals.
IMPUTER_PARAMS_REG_EVAL = {
    "imputer_choice": "extra_trees",
    "et_estimators": 20,
    "et_max_depth": 5,
    "extra_trees_max_iter": 6,
    "initial_strategy": "median",
}

# Used for both tuning and evaluation of the classification models.
IMPUTER_PARAMS_CLF = {
    "imputer_choice": "mice",
    "mice_max_iter": 6,
    "mice_initial_strategy": "median",
    "mice_et_estimators": 19,
    "mice_et_max_depth": 5,
}

# Fraction of already-observed cells hidden when scoring imputers against
# known ground truth, and the number of Optuna trials for the imputer search.
IMPUTER_MASK_FRACTION = 0.10
IMPUTER_STUDY_TRIALS = 100
IMPUTER_STUDY_N_JOBS = 4

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
N_SPLITS = 5     # StratifiedKFold folds used by every tuning objective
TEST_SIZE = 0.2  # hold-out fraction of the train/test split

# ---------------------------------------------------------------------------
# Optuna study configuration — trial budget per model
# ---------------------------------------------------------------------------
# Both tracks maximise their metric: QWK for regression, Macro F1 for
# classification.
STUDY_TRIALS = {
    "regression": {
        "lgbm": 50, "xgb": 20, "cb": 20, "rf": 20,
        "ridge": 20, "svr": 20, "nn": 20, "lasso": 20,
    },
    "classification": {
        "lgbm": 50, "xgb": 50, "cb": 50, "rf": 50,
        "ridge": 100, "svc": 50, "lr": 100, "nn": 100,
    },
}

# ---------------------------------------------------------------------------
# Fixed (non-tuned) model keyword arguments used at FINAL EVALUATION time
# ---------------------------------------------------------------------------
# These are merged on top of the Optuna-tuned parameters when the final models
# are rebuilt. They reproduce exactly the keyword arguments that the original
# evaluation notebooks passed inline.
EVAL_FIXED_PARAMS_REG = {
    "LightGBM": {"random_state": 42, "verbose": -1},
    "XGBoost": {"random_state": 42, "verbosity": 0},
    "CatBoost": {"random_seed": 42, "verbose": 0, "allow_writing_files": False},
    "RandomForest": {"random_state": 42, "n_jobs": -1},
    "Ridge": {"random_state": 42},
    "Lasso": {"random_state": 42},
    "MLP": {},  # device / verbose are injected when the MLP config is loaded
}

EVAL_FIXED_PARAMS_CLF = {
    "LightGBM": {"objective": "multiclass", "random_state": 42,
                 "verbose": -1, "class_weight": "balanced"},
    "XGBoost": {"objective": "multi:softprob", "random_state": 42,
                "verbosity": 0},
    "CatBoost": {"loss_function": "MultiClass", "random_seed": 42,
                 "verbose": False, "auto_class_weights": "Balanced"},
    "RandomForest": {"random_state": 42, "n_jobs": -1,
                     "class_weight": "balanced"},
    "RidgeClassifier": {"random_state": 42, "class_weight": "balanced"},
    "LogisticRegression": {"random_state": 42, "class_weight": "balanced"},
    "SVC": {"probability": True, "random_state": 42,
            "class_weight": "balanced"},
    "MLP": {"epochs": 100, "device": "cpu"},
}

# The classification stacking ensemble rebuilds four of its base learners with
# a lighter set of fixed arguments than the standalone evaluation above — most
# notably the MLP trains for 50 epochs instead of 100, because the stacker fits
# it five times over during its internal cross-validation.
STACK_FIXED_PARAMS_CLF = {
    "rf": {"random_state": 42, "class_weight": "balanced"},
    "xgb": {"random_state": 42},
    "svc": {"probability": True, "random_state": 42,
            "class_weight": "balanced"},
    "nn": {"device": "cpu", "epochs": 50},
}

# ---------------------------------------------------------------------------
# Columns dropped from the raw dataset
# ---------------------------------------------------------------------------
# The target `sii` is computed directly from the PCIAT questionnaire, so every
# PCIAT column has to go or the models simply read the answer off the input
# (a naive model that keeps them scores QWK ~0.99). CGAS and the Season columns
# are dropped for the same leakage reason.
#
# This is a schema-level decision — it does not depend on any row's values — so
# it is safe to apply *before* the train/test split.

LEAKAGE_COLS = ["CGAS-CGAS_Score"]

SEASON_COLS = [
    "CGAS-Season", "Physical-Season", "Fitness_Endurance-Season",
    "FGC-Season", "BIA-Season", "PAQ_A-Season", "PAQ_C-Season",
    "PCIAT-Season", "Basic_Demos-Enroll_Season",
]

# The 20 individual PCIAT questionnaire items: PCIAT-PCIAT_01 ... _20.
PCIAT_ITEM_COLS = [f"PCIAT-PCIAT_{i:02d}" for i in range(1, 21)]

# Full set removed before the split.
FEATURE_DROP_COLS = LEAKAGE_COLS + SEASON_COLS + PCIAT_ITEM_COLS

# Columns removed from the *processed* CSVs when separating features from the
# target. Note that the stages disagree on `is_outlier`, on purpose — see
# data_loader.load_processed_split() for the details.
META_COLS = ["sii", "Split", "id", "is_outlier"]

# ---------------------------------------------------------------------------
# Consensus outlier detection
# ---------------------------------------------------------------------------
# A row is called an outlier only if at least OUTLIER_MIN_VOTES of the ten
# detectors put it above their own OUTLIER_PERCENTILE-th score percentile.
OUTLIER_PERCENTILE = 99   # used for the final consensus flag (top 1% per method)
OUTLIER_COMPARE_PERCENTILE = 90  # used for the method-agreement diagnostic plot
OUTLIER_MIN_VOTES = 9

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
SII_CLASSES = [0, 1, 2, 3]
SII_CLASS_NAMES = [
    "sii=0 (None)", "sii=1 (Mild)", "sii=2 (Moderate)", "sii=3 (Severe)",
]
