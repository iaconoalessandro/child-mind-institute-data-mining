"""
STEP 2 & 3 — Data loading and preprocessing.

The flow implemented here, in order:

    load_raw_dataset()        read cmi_internet.csv
    audit_data()              blank out physically impossible measurements
    drop_leakage_columns()    remove PCIAT / CGAS / Season columns
    split_train_test()        stratified 80/20 split on the sii target
    apply_train_only_filters() drop mostly-empty and near-duplicate features
    save_processed_split()    write the result to dataset/PostProcessed_*.csv

Later stages read those CSVs back with load_processed_split() and turn them
into model-ready matrices with preprocess_split().

The ordering matters. Everything before the split is a *schema* decision that
does not look at any row's values, so it cannot leak test information. The
filters after the split are data-dependent, so they are computed on the
training rows only and then applied to the test rows.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (FEATURE_DROP_COLS, META_COLS, PATHS, SEEDS, TEST_SIZE)

# Column sets dropped when a processed CSV is turned back into (X, y).
# The three stages deliberately differ, and the difference is preserved
# because it is baked into the reported results:
#   * tuning keeps `is_outlier` as an ordinary feature
#   * final regression evaluation drops it along with the other meta columns
DROP_FOR_TUNING = ["sii", "Split"]
DROP_FOR_REG_EVAL = META_COLS
DROP_FOR_CLASSIFICATION = ["sii", "Split", "PCIAT-PCIAT_Total"]

# Plausible ranges for physical measurements. Anything outside these is a
# data-entry or instrument error rather than a real reading, so we replace it
# with NaN and let the imputer fill it in. `None` means "no bound on that side".
PLAUSIBLE_RANGES = {
    "Basic_Demos-Age": (5, 22),          # study recruited ages 5-22
    "BIA-BIA_BMI": (10, 50),
    "Physical-HeartRate": (30, 220),     # bpm
    "Physical-Systolic_BP": (50, 200),   # mmHg
    "Physical-Diastolic_BP": (30, 130),  # mmHg
    "Physical-Height": (30, None),       # inches
    "Physical-Weight": (10, None),       # pounds
}


# ---------------------------------------------------------------------------
# Loading and cleaning the raw dataset
# ---------------------------------------------------------------------------

def load_raw_dataset(path=None):
    """Read the raw Child Mind Institute CSV into a DataFrame.

    Args:
        path: Optional override for the CSV location.

    Returns:
        pd.DataFrame with one row per participant.
    """
    return pd.read_csv(path or PATHS["raw_data"])


def audit_data(dataframe):
    """Replace physically impossible measurements with NaN.

    Args:
        dataframe: Raw dataset.

    Returns:
        A new DataFrame; the input is not modified.
    """
    audited = dataframe.copy()
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column not in audited.columns:
            continue
        out_of_range = audited[column] < low
        if high is not None:
            out_of_range = out_of_range | (audited[column] > high)
        audited.loc[out_of_range, column] = np.nan
    return audited


def drop_leakage_columns(dataframe):
    """Remove target-leaking columns and keep only the numeric ones.

    `sii` is derived directly from the PCIAT questionnaire, so leaving any
    PCIAT column in the features lets a model read the answer off its input.
    See ``config.FEATURE_DROP_COLS`` for the exact list and the reasoning.

    Args:
        dataframe: Audited dataset.

    Returns:
        pd.DataFrame with only numeric, non-leaking columns.
    """
    present = [c for c in FEATURE_DROP_COLS if c in dataframe.columns]
    return dataframe.drop(columns=present, errors="ignore").select_dtypes(
        include=[np.number]
    )


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_train_test(numeric_dataframe):
    """Stratified 80/20 split over every row that has a known `sii` label.

    ``PCIAT-PCIAT_Total`` and ``id`` are dropped from the features here:
    the former is the raw score `sii` is thresholded from, the latter is a
    participant identifier with no predictive content.

    Args:
        numeric_dataframe: Output of :func:`drop_leakage_columns`.

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)``, all with a fresh
        0..n-1 index.
    """
    labelled = numeric_dataframe[numeric_dataframe["sii"].notna()].copy()
    features = labelled.drop(
        columns=["PCIAT-PCIAT_Total", "id", "sii"], errors="ignore"
    )
    target = labelled["sii"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=TEST_SIZE,
        random_state=SEEDS["train_test_split"],
        stratify=target,
    )
    return (X_train.reset_index(drop=True), X_test.reset_index(drop=True),
            y_train.reset_index(drop=True), y_test.reset_index(drop=True))


def apply_train_only_filters(X_train, X_test, label="", max_missing=0.5,
                             max_abs_spearman=0.90):
    """Drop weak / redundant features, deciding only from the training rows.

    Two filters run in sequence:

    1. Columns missing more than ``max_missing`` of their training values are
       dropped — too little signal to impute reliably.
    2. For every pair of columns whose absolute Spearman correlation exceeds
       ``max_abs_spearman``, the later of the two is dropped. Highly redundant
       features inflate the dimensionality without adding information.

    Both thresholds are measured on ``X_train`` only, then the same columns are
    removed from ``X_test``, so no test row influences which features survive.

    Args:
        X_train: Training features.
        X_test: Test features.
        label: Prefix used in the progress printout, e.g. ``"Regression"``.
        max_missing: Missing-value fraction above which a column is dropped.
        max_abs_spearman: |rho| above which one of a correlated pair is dropped.

    Returns:
        Tuple ``(X_train_filtered, X_test_filtered)`` with identical columns.
    """
    tag = f"[{label}] " if label else ""

    missing_fraction = X_train.isnull().mean()
    mostly_empty = missing_fraction[missing_fraction > max_missing].index.tolist()
    print(f"{tag}Dropping {len(mostly_empty)} columns with "
          f">{max_missing:.0%} missing on TRAIN: {mostly_empty}")
    X_train = X_train.drop(columns=mostly_empty)
    X_test = X_test.drop(columns=mostly_empty)

    # Look only at the upper triangle so each pair is considered once, and
    # drop the column that owns the offending cell.
    correlation = X_train.corr(method="spearman").abs()
    upper_triangle = correlation.where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )
    redundant = [c for c in upper_triangle.columns
                 if any(upper_triangle[c] > max_abs_spearman)]
    print(f"{tag}Dropping {len(redundant)} columns with Spearman "
          f"|rho| > {max_abs_spearman} on TRAIN: {redundant}")
    X_train = X_train.drop(columns=redundant)
    X_test = X_test.drop(columns=redundant)

    # Guarantee the two matrices present their columns in the same order.
    X_test = X_test.reindex(columns=X_train.columns)

    print(f"Final train shape: {X_train.shape} | test shape: {X_test.shape}")
    print(f"Feature columns aligned: {X_train.columns.equals(X_test.columns)}")
    return X_train, X_test


# ---------------------------------------------------------------------------
# Reading and writing the processed datasets
# ---------------------------------------------------------------------------

def save_processed_split(X_train, y_train, X_test, y_test, outlier_mask,
                         output_path):
    """Write train and test rows to one CSV, tagged by a `Split` column.

    Test rows always get ``is_outlier = 0``: outlier detection is a
    training-set diagnostic and is never run on held-out data.

    Args:
        X_train: Filtered training features.
        y_train: Training labels.
        X_test: Filtered test features.
        y_test: Test labels.
        outlier_mask: Boolean array flagging consensus outliers in the train set.
        output_path: Destination CSV path.

    Returns:
        The combined pd.DataFrame that was written.
    """
    train_rows = X_train.copy()
    train_rows["sii"] = np.asarray(y_train).astype(int)
    train_rows["is_outlier"] = np.asarray(outlier_mask).astype(int)
    train_rows["Split"] = "Train"

    test_rows = X_test.copy()
    test_rows["sii"] = np.asarray(y_test).astype(int)
    test_rows["is_outlier"] = 0
    test_rows["Split"] = "Test"

    combined = pd.concat([train_rows, test_rows], ignore_index=True)
    combined.to_csv(output_path, index=False)

    print(f"Rows written -> Train: {len(train_rows)}, Test: {len(test_rows)}")
    print(f"Outliers flagged in Train: {train_rows['is_outlier'].sum()} "
          f"({100 * train_rows['is_outlier'].mean():.2f}%)")
    return combined


def validate_processed_file(path):
    """Re-read a processed CSV and print basic sanity checks.

    Confirms the `Split` column only holds Train/Test, that `sii` is complete
    and within {0,1,2,3}, and that train and test expose the same features.

    Args:
        path: Path to a processed CSV.

    Returns:
        True if every check passes.
    """
    saved = pd.read_csv(path)
    train_part = saved[saved["Split"] == "Train"]
    test_part = saved[saved["Split"] == "Test"]

    split_values = sorted(saved["Split"].dropna().unique().tolist())
    split_ok = set(split_values).issubset({"Train", "Test"})
    sii_complete = saved["sii"].notna().all()
    sii_values = sorted(saved["sii"].astype(int).unique().tolist())
    sii_ok = set(sii_values).issubset({0, 1, 2, 3})

    meta = ["sii", "Split", "is_outlier"]
    train_features = [c for c in train_part.columns if c not in meta]
    test_features = [c for c in test_part.columns if c not in meta]
    columns_match = train_features == test_features

    print(f"Validation - Split values only Train/Test: {split_ok} -> {split_values}")
    print(f"Validation - sii non-null: {sii_complete}")
    print(f"Validation - sii classes in {{0,1,2,3}}: {sii_ok} -> {sii_values}")
    print(f"Validation - feature columns identical: {columns_match}")
    return bool(split_ok and sii_complete and sii_ok and columns_match)


def load_processed_split(path, drop_columns):
    """Read a processed CSV back into train/test feature and label pairs.

    Args:
        path: Path to a processed CSV written by :func:`save_processed_split`.
        drop_columns: Columns to remove from the features. Use one of the
            ``DROP_FOR_*`` constants at the top of this module — they encode
            the per-stage differences deliberately.

    Returns:
        Tuple ``(X_train, y_train, X_test, y_test)``.
    """
    processed = pd.read_csv(path)
    train_rows = processed[processed["Split"] == "Train"].copy()
    test_rows = processed[processed["Split"] == "Test"].copy()

    def features_of(rows):
        present = [c for c in drop_columns if c in rows.columns]
        return rows.drop(columns=present).reset_index(drop=True)

    return (features_of(train_rows),
            train_rows["sii"].astype(int).reset_index(drop=True),
            features_of(test_rows),
            test_rows["sii"].astype(int).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def build_imputer(imputer_params, add_indicator=False):
    """Rebuild an imputer from a tuned parameter dictionary.

    ``imputer_params["imputer_choice"]`` selects the strategy; the remaining
    keys are that strategy's hyperparameters. Values are wrapped in
    ``int(float(...))`` because they may arrive as strings or floats after a
    CSV round-trip.

    Note the ``initial_strategy`` lookup: the tuned dictionaries in
    ``config.py`` store it as ``mice_initial_strategy`` / ``et_initial_strategy``
    rather than ``initial_strategy``, so most configurations fall back to
    scikit-learn's ``"mean"`` default. That is the behaviour the published
    results were produced with — see the note in ``config.py``.

    Args:
        imputer_params: Dict with ``imputer_choice`` plus strategy-specific keys.
        add_indicator: Whether to append binary missing-value indicator columns.

    Returns:
        An unfitted ``IterativeImputer`` or ``KNNImputer``.
    """
    choice = imputer_params.get("imputer_choice", "lasso")
    initial_strategy = imputer_params.get("initial_strategy", "mean")
    seed = SEEDS["random_state"]

    if choice == "knn":
        return KNNImputer(
            n_neighbors=int(float(imputer_params.get("n_neighbors", 5))),
            add_indicator=add_indicator,
        )

    # The tree-based strategies share one shape and differ only in which
    # estimator they round-robin over the columns, and which keys hold its size.
    tree_estimators = {
        "missforest": (RandomForestRegressor, "rf_n_estimators", "rf_max_depth"),
        "mice": (ExtraTreesRegressor, "mice_et_estimators", "mice_et_max_depth"),
        "extra_trees": (ExtraTreesRegressor, "et_estimators", "et_max_depth"),
    }
    if choice in tree_estimators:
        estimator_class, n_estimators_key, max_depth_key = tree_estimators[choice]

        raw_max_depth = imputer_params.get(max_depth_key)
        max_depth = (int(float(raw_max_depth))
                     if raw_max_depth is not None and pd.notnull(raw_max_depth)
                     else None)

        return IterativeImputer(
            estimator=estimator_class(
                n_estimators=int(float(imputer_params.get(n_estimators_key, 100))),
                max_depth=max_depth,
                random_state=seed,
                n_jobs=-1,
            ),
            initial_strategy=initial_strategy,
            add_indicator=add_indicator,
            max_iter=int(float(imputer_params.get(f"{choice}_max_iter", 10))),
            random_state=seed,
        )

    # The linear strategies differ only in the per-column regressor.
    if choice == "lasso":
        estimator = Lasso(alpha=float(imputer_params.get("alpha", 0.1)),
                          random_state=seed)
    elif choice == "ridge":
        estimator = Ridge(alpha=float(imputer_params.get("alpha_ridge", 1.0)),
                          random_state=seed)
    elif choice == "logistic":
        estimator = LogisticRegression(
            C=float(imputer_params.get("C_logistic", 1.0)),
            random_state=seed, max_iter=1000,
        )
    elif choice == "bayesian_ridge":
        estimator = BayesianRidge(
            alpha_1=float(imputer_params.get("br_alpha_1", 1e-6)),
            alpha_2=float(imputer_params.get("br_alpha_2", 1e-6)),
            lambda_1=float(imputer_params.get("br_lambda_1", 1e-6)),
            lambda_2=float(imputer_params.get("br_lambda_2", 1e-6)),
        )
    else:
        # Unrecognised choice: fall back to a plain Lasso imputer. This path
        # keeps scikit-learn's default initial_strategy rather than the one
        # resolved above, matching the original implementation.
        return IterativeImputer(
            estimator=Lasso(alpha=imputer_params.get("alpha", 1.0),
                            random_state=seed),
            add_indicator=add_indicator,
            random_state=seed,
        )

    return IterativeImputer(
        estimator=estimator,
        initial_strategy=initial_strategy,
        add_indicator=add_indicator,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# Preprocessing pipelines
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(imputer_params, n_components=None):
    """Assemble the standard preprocessing pipeline.

    Order is StandardScaler -> imputer -> optional PCA. Scaling first keeps the
    iterative imputer's internal regressions on a common numeric scale.

    Args:
        imputer_params: Passed straight to :func:`build_imputer`.
        n_components: If given, a PCA step with this many components is added.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.
    """
    steps = [
        ("scaler", StandardScaler()),
        ("imputer", build_imputer(imputer_params, add_indicator=False)),
    ]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components,
                                 random_state=SEEDS["pca"])))
    return Pipeline(steps)


def preprocess_split(X_fit, X_apply, y_fit, imputer_params, n_components=None):
    """Fit preprocessing on one set of rows and apply it to another.

    This is the single guard against leakage during model selection: the
    scaler's means, the imputer's regressions and the PCA basis are all learned
    from ``X_fit`` alone and then merely *applied* to ``X_apply``.

    Use it both for a single cross-validation fold (fit = train fold, apply =
    validation fold) and for the final refit (fit = full train, apply = test).

    Args:
        X_fit: Rows the pipeline learns from.
        X_apply: Rows the fitted pipeline transforms.
        y_fit: Labels for ``X_fit``, returned with a reset index for alignment.
        imputer_params: Passed to :func:`build_preprocessing_pipeline`.
        n_components: PCA component count, or None for no PCA.

    Returns:
        Tuple ``(X_fit_transformed, X_apply_transformed, y_fit, pipeline)``.
    """
    pipeline = build_preprocessing_pipeline(imputer_params, n_components)
    X_fit_transformed = pipeline.fit_transform(X_fit)
    X_apply_transformed = pipeline.transform(X_apply)

    if hasattr(y_fit, "reset_index"):
        y_fit = y_fit.reset_index(drop=True)

    return X_fit_transformed, X_apply_transformed, y_fit, pipeline
