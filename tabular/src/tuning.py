"""
STEP 4a — Hyperparameter tuning with Optuna.

Each of the sixteen studies (eight regressors, eight classifiers) works the
same way:

    1. Optuna proposes a set of hyperparameters (a "trial").
    2. We run 5-fold stratified cross-validation with those settings.
    3. The mean fold score is handed back, and Optuna proposes a better trial.

The important detail is inside :func:`evaluate_fold`: the scaler, imputer and
PCA are re-fitted on each training fold and only *applied* to the validation
fold. Fitting them once on the whole training set would let information from
the validation rows reach the model and inflate every score.

Search spaces live in the ``suggest_*`` functions. The order of the
``trial.suggest_*`` calls inside them is significant — Optuna's sampler keys
off it — so it is preserved exactly as originally written.
"""

import os
import warnings

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (Lasso, LogisticRegression, Ridge,
                                  RidgeClassifier)
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_sample_weight

from src.config import N_SPLITS, SEEDS, STUDY_TRIALS
from src.data_loader import preprocess_split
from src.metrics import apply_thresholds, optimize_thresholds_grid
from src.models import TorchClassifier, TorchRegressor
from src.utils import get_n_jobs

# These fire constantly from linear models on partially imputed folds and
# would otherwise bury the progress bar.
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ---------------------------------------------------------------------------
# Saving study progress
# ---------------------------------------------------------------------------

class SaveCallback:
    """Write the study's trials to CSV after every trial.

    Optuna studies here are in-memory, so a crashed or interrupted run would
    otherwise lose everything. Any pre-existing CSV is loaded on construction
    and new trials are appended to it, so history survives across restarts.

    Args:
        path: Destination CSV path.
    """

    def __init__(self, path):
        self.path = path
        self.previous_trials = pd.read_csv(path) if os.path.exists(path) else None

    def __call__(self, study, trial):
        current = study.trials_dataframe()
        if self.previous_trials is not None:
            current = pd.concat([self.previous_trials, current],
                                ignore_index=True)
        current.to_csv(self.path, index=False)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def evaluate_fold(train_idx, valid_idx, X_raw, y, params, model_class,
                  imputer_params, task="reg", n_components=None):
    """Train on one fold and score it on the held-out fold.

    Args:
        train_idx: Row positions of the training fold.
        valid_idx: Row positions of the validation fold.
        X_raw: Un-preprocessed training features (a DataFrame).
        y: Training labels (a Series).
        params: Hyperparameters for this trial.
        model_class: Estimator class to instantiate.
        imputer_params: Imputer configuration for the fold pipeline.
        task: ``"reg"`` to score threshold-optimised QWK, ``"clf"`` for Macro F1.
        n_components: PCA component count, or None.

    Returns:
        The fold's score as a float.
    """
    # Re-declared inside the function because joblib workers are fresh
    # processes that do not inherit the module-level warning filters.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    y_train_fold = y.iloc[train_idx]
    y_valid_fold = y.iloc[valid_idx]

    X_train_fold, X_valid_fold, y_train_fold, _ = preprocess_split(
        X_raw.iloc[train_idx], X_raw.iloc[valid_idx], y_train_fold,
        imputer_params, n_components=n_components,
    )

    # Weight rare classes up so the model is not rewarded for ignoring them.
    sample_weight = compute_sample_weight("balanced", y_train_fold)

    model = model_class(**params)
    try:
        model.fit(X_train_fold, y_train_fold, sample_weight=sample_weight)
    except TypeError:
        # Some estimators want `verbose` alongside sample_weight; others accept
        # no sample weighting at all and handle imbalance internally.
        try:
            model.fit(X_train_fold, y_train_fold,
                      sample_weight=sample_weight, verbose=False)
        except Exception:
            model.fit(X_train_fold, y_train_fold)

    if task == "reg":
        # Continuous output has to be cut into classes first. The cut points
        # are fitted on the training fold's predictions, never the validation
        # fold's, then applied to the validation predictions.
        try:
            thresholds = optimize_thresholds_grid(y_train_fold,
                                                  model.predict(X_train_fold))
            class_predictions = apply_thresholds(model.predict(X_valid_fold),
                                                 thresholds)
        except Exception:
            # Fall back to plain rounding if the threshold search fails.
            class_predictions = np.clip(
                np.round(model.predict(X_valid_fold)), 0, 3
            ).astype(int)
        return cohen_kappa_score(y_valid_fold, class_predictions,
                                 weights="quadratic")

    class_predictions = np.asarray(model.predict(X_valid_fold)).ravel()
    return f1_score(y_valid_fold, class_predictions, average="macro")


def cross_validated_score(params, model_class, X_raw, y, imputer_params, task,
                          n_components=None):
    """Average the fold scores of a 5-fold stratified cross-validation.

    Stratified folds keep the rare severe cases represented in every split.

    Args:
        params: Hyperparameters for this trial.
        model_class: Estimator class to instantiate.
        X_raw: Un-preprocessed training features.
        y: Training labels.
        imputer_params: Imputer configuration for the fold pipelines.
        task: ``"reg"`` or ``"clf"``.
        n_components: PCA component count, or None.

    Returns:
        Mean fold score as a float.
    """
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                               random_state=SEEDS["cv_shuffle"])
    fold_scores = Parallel(n_jobs=get_n_jobs())(
        delayed(evaluate_fold)(train_idx, valid_idx, X_raw, y, params,
                               model_class, imputer_params, task, n_components)
        for train_idx, valid_idx in splitter.split(X_raw, y)
    )
    return float(np.mean(fold_scores))


def compare_resampling_strategies(X_train, y_train):
    """Benchmark seven class-imbalance strategies with a fast LightGBM proxy.

    Every resampler sits *inside* an imblearn pipeline, so it is refitted
    within each cross-validation fold. Resampling the whole training set before
    splitting would put synthetic copies of validation rows into the training
    folds and produce meaningless scores.

    This is a diagnostic. The pipeline does not resample in the end — the
    tuned models use ``class_weight="balanced"`` instead, which reweights the
    loss rather than duplicating rows.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        pd.DataFrame of mean Macro F1, recall and precision per strategy.
    """
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import ADASYN, SMOTE
    from imblearn.pipeline import Pipeline as ImbalancedPipeline
    from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import precision_score, recall_score
    from sklearn.preprocessing import StandardScaler

    seed = SEEDS["random_state"]
    proxy_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5,
                                     random_state=seed, n_jobs=-1, verbose=-1)
    strategies = {
        "Baseline (No Resampling)": "passthrough",
        "SMOTE": SMOTE(random_state=seed),
        "ADASYN": ADASYN(random_state=seed),
        "ENN": EditedNearestNeighbours(),
        "Tomek Links": TomekLinks(),
        "SMOTE + ENN": SMOTEENN(random_state=seed),
        "SMOTE + Tomek": SMOTETomek(random_state=seed),
    }

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                               random_state=SEEDS["cv_shuffle"])
    print(f"{'Method':<25} | {'Macro F1':<10} | {'Recall':<10} | {'Precision':<10}")
    print("-" * 65)

    rows = []
    for name, resampler in strategies.items():
        pipeline = ImbalancedPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("resample", resampler),
            ("classifier", proxy_model),
        ])

        f1_scores, recalls, precisions = [], [], []
        for train_idx, valid_idx in splitter.split(X_train, y_train):
            pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            y_valid = y_train.iloc[valid_idx]
            predictions = pipeline.predict(X_train.iloc[valid_idx])
            f1_scores.append(f1_score(y_valid, predictions, average="macro"))
            recalls.append(recall_score(y_valid, predictions, average="macro"))
            precisions.append(precision_score(y_valid, predictions,
                                              average="macro"))

        rows.append({"Method": name, "Macro F1": np.mean(f1_scores),
                     "Recall": np.mean(recalls),
                     "Precision": np.mean(precisions)})
        print(f"{name:<25} | {rows[-1]['Macro F1']:<10.4f} | "
              f"{rows[-1]['Recall']:<10.4f} | {rows[-1]['Precision']:<10.4f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
# Every function takes ``(trial, n_features, task)`` and returns
# ``(params, n_components)``, so ``tune_model`` below can call any of them the
# same way. Some ignore ``task`` because their search space is identical for
# regression and classification.
#
# ``n_components`` is None for models fed the full feature set, and an integer
# for the models that are tuned together with a PCA compression step.
#
# The models without PCA are the tree ensembles, which cope with many
# correlated features on their own. The linear models, SVMs and the MLP all
# benefit from the compression, so the component count is tuned alongside
# their own hyperparameters.

def suggest_lgbm(trial, n_features, task):
    """LightGBM gradient-boosted trees."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    if task == "reg":
        params.update({"objective": "regression", "metric": "rmse",
                       "boosting_type": "gbdt",
                       "random_state": SEEDS["random_state"], "verbose": -1})
    else:
        params.update({"objective": "multiclass", "metric": "multi_logloss",
                       "boosting_type": "gbdt",
                       "random_state": SEEDS["random_state"], "verbose": -1,
                       "class_weight": "balanced"})
    return params, None


def suggest_xgb(trial, n_features, task):
    """XGBoost gradient-boosted trees."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
    }
    if task == "reg":
        params.update({"objective": "reg:squarederror",
                       "random_state": SEEDS["random_state"], "verbosity": 0})
    else:
        params.update({"objective": "multi:softprob", "num_class": 4,
                       "random_state": SEEDS["random_state"], "verbosity": 0})
    return params, None


def suggest_catboost(trial, n_features, task):
    """CatBoost gradient-boosted trees."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "iterations": trial.suggest_int("iterations", 50, 300),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }
    shared = {"random_seed": SEEDS["catboost_seed"], "verbose": False,
              "allow_writing_files": False, "thread_count": 4}
    if task == "reg":
        params.update({"loss_function": "RMSE", **shared})
    else:
        params.update({"loss_function": "MultiClass", **shared,
                       "auto_class_weights": "Balanced"})
    return params, None


def suggest_random_forest(trial, n_features, task):
    """Random Forest."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
    }
    params.update({"random_state": SEEDS["random_state"], "n_jobs": -1})
    if task == "clf":
        params["class_weight"] = "balanced"
    return params, None


def suggest_ridge(trial, n_features, task):
    """Ridge regression / classifier, with PCA compression."""
    params = {"alpha": trial.suggest_float("alpha", 0.1, 100.0, log=True),
              "random_state": SEEDS["random_state"]}
    if task == "clf":
        params["class_weight"] = "balanced"
    n_components = trial.suggest_int("pca_n_components", 5, min(n_features, 100))
    return params, n_components


def suggest_lasso(trial, n_features, task):
    """Lasso regression, with PCA compression."""
    params = {"alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True),
              "random_state": SEEDS["random_state"]}
    n_components = trial.suggest_int("pca_n_components", 5, min(n_features, 100))
    return params, n_components


def suggest_svm(trial, n_features, task):
    """Support vector machine, with PCA compression.

    ``degree`` only means anything for a polynomial kernel, so it is suggested
    conditionally and pinned to 3 otherwise.
    """
    kernel = trial.suggest_categorical("kernel",
                                       ["linear", "poly", "rbf", "sigmoid"])
    degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3

    params = {
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "kernel": kernel,
        "degree": degree,
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }
    if task == "reg":
        params["epsilon"] = trial.suggest_float("epsilon", 0.01, 1.0, log=True)
    else:
        params.update({"random_state": SEEDS["random_state"],
                       "probability": True, "class_weight": "balanced"})

    n_components = trial.suggest_int("pca_n_components", 5, min(n_features, 100))
    return params, n_components


def suggest_logistic_regression(trial, n_features, task):
    """Logistic regression, with PCA compression."""
    params = {
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l2"]),
        "solver": trial.suggest_categorical(
            "solver", ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"]),
        "random_state": SEEDS["random_state"],
        "class_weight": "balanced",
    }
    n_components = trial.suggest_int("pca_n_components", 5, min(n_features, 100))
    return params, n_components


def suggest_mlp(trial, n_features, task):
    """PyTorch MLP, with PCA compression.

    The architecture is searched as a depth plus one width per layer. PCA is
    capped lower here (45 components) than for the linear models, because a
    network with this little data overfits quickly on a wide input.
    """
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layers = [trial.suggest_int(f"n_units_l{layer}", 32, 256)
                     for layer in range(1, n_layers + 1)]

    params = {
        "hidden_layers": hidden_layers,
        "activation": trial.suggest_categorical(
            "activation", ["gelu", "silu", "leaky_relu"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 50, 500),
        "batch_size": trial.suggest_categorical("batch_size",
                                                [32, 64, 128, 256, 512]),
        "device": "mps",
        "verbose": False,
        "optimizer_name": trial.suggest_categorical(
            "optimizer", ["adamw", "adam", "sgd", "rmsprop"]),
        # Recorded so the study is comparable with earlier runs; the estimator
        # ignores it (it trains every parameter with a single optimizer).
        "optimizer_secondary": trial.suggest_categorical(
            "optimizer_secondary", ["adam", "sgd", "adamw", "rmsprop"]),
    }

    n_components = trial.suggest_int("pca_n_components", 3, min(n_features, 45))
    return params, n_components


# Model key -> (estimator class, search-space function). The key also names the
# study's output CSV, e.g. "lgbm" -> optuna trials/lgbm_reg_trials.csv.
REGRESSION_MODELS = {
    "lgbm": (lgb.LGBMRegressor, suggest_lgbm),
    "xgb": (xgb.XGBRegressor, suggest_xgb),
    "cb": (cb.CatBoostRegressor, suggest_catboost),
    "rf": (RandomForestRegressor, suggest_random_forest),
    "ridge": (Ridge, suggest_ridge),
    "svr": (SVR, suggest_svm),
    "nn": (TorchRegressor, suggest_mlp),
    "lasso": (Lasso, suggest_lasso),
}

CLASSIFICATION_MODELS = {
    "lgbm": (lgb.LGBMClassifier, suggest_lgbm),
    "xgb": (xgb.XGBClassifier, suggest_xgb),
    "cb": (cb.CatBoostClassifier, suggest_catboost),
    "rf": (RandomForestClassifier, suggest_random_forest),
    "ridge": (RidgeClassifier, suggest_ridge),
    "svc": (SVC, suggest_svm),
    "lr": (LogisticRegression, suggest_logistic_regression),
    "nn": (TorchClassifier, suggest_mlp),
}

# Human-readable study names, used for Optuna's study_name only.
STUDY_NAMES = {
    "lgbm": "LightGBM", "xgb": "XGBoost", "cb": "CatBoost",
    "rf": "RandomForest", "ridge": "Ridge", "lasso": "Lasso",
    "svr": "SVR", "svc": "SVC", "lr": "LogisticRegression", "nn": "TorchMLP",
}


# ---------------------------------------------------------------------------
# Running a study
# ---------------------------------------------------------------------------

def tune_model(model_key, task, X_train_raw, y_train, imputer_params,
               results_dir, n_trials=None):
    """Run one Optuna study and save its trials to CSV.

    Args:
        model_key: Key into ``REGRESSION_MODELS`` / ``CLASSIFICATION_MODELS``.
        task: ``"reg"`` or ``"clf"``.
        X_train_raw: Un-preprocessed training features.
        y_train: Training labels.
        imputer_params: Imputer configuration used inside every fold.
        results_dir: Directory the trials CSV is written to.
        n_trials: Trial budget; defaults to the value in ``config.STUDY_TRIALS``.

    Returns:
        The completed ``optuna.Study``.
    """
    registry = REGRESSION_MODELS if task == "reg" else CLASSIFICATION_MODELS
    model_class, suggest_params = registry[model_key]
    metric_name = "QWK" if task == "reg" else "Macro F1"
    track = "regression" if task == "reg" else "classification"
    n_trials = n_trials or STUDY_TRIALS[track][model_key]

    print(f"Starting {STUDY_NAMES[model_key]} optimization "
          f"(maximizing {metric_name})...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params, n_components = suggest_params(trial, X_train_raw.shape[1], task)
        return cross_validated_score(params, model_class, X_train_raw, y_train,
                                     imputer_params, task, n_components)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{STUDY_NAMES[model_key]}_{task.capitalize()}_Tuning",
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
        callbacks=[SaveCallback(
            os.path.join(results_dir, f"{model_key}_{task}_trials.csv"))],
    )

    print(f"{STUDY_NAMES[model_key]}: Best trial CV {metric_name}: "
          f"{study.best_value:.4f}")
    return study
