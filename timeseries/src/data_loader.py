"""
STEP 2 — Data loading and preparation.

Turns the raw pickle of 4,437 per-subject DataFrames into the four NumPy arrays
every later stage reads. The chain is:

    load -> stratified split -> Hampel outlier filter -> interpolate ->
    Z-score normalise -> stack and save

Two asymmetries between train and test are deliberate and are preserved here
exactly as the original analysis ran them:

* Only the **training** subjects go through the Hampel filter and the
  interpolation that fills the holes it leaves. The test subjects keep their
  raw values.
* The scalers are fitted on training rows only, then applied to both halves,
  so no test row influences the normalisation.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.outlier_detection import HampelFilter

from src.config import (COLUMNS_TO_DROP, HAMPEL_CONFIGS, HAMPEL_DEFAULT,
                        HAMPEL_SKIP_COLUMNS, PATHS, SCALED_FEATURES, SEEDS,
                        TEST_SIZE)


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def load_raw_dataset():
    """Read the pickled list of per-subject DataFrames.

    Returns:
        List of DataFrames, one per subject. Each has 200 rows (30-minute time
        steps) and 13 columns: the sensor channels plus `id` and `sii_binary`.
    """
    with open(PATHS["raw_data"], "rb") as handle:
        return pickle.load(handle)


def build_subject_metadata(dataset):
    """Collect one row per subject holding just the label used for splitting.

    Each subject's `sii_binary` is constant down all 200 rows, so reading the
    first row is enough to know which class the subject belongs to.

    Args:
        dataset: List of per-subject DataFrames.

    Returns:
        DataFrame with columns ``original_index`` and ``sii_binary``.
    """
    return pd.DataFrame([
        {"original_index": index, "sii_binary": subject["sii_binary"].iloc[0]}
        for index, subject in enumerate(dataset)
    ])


def split_subjects(subject_metadata):
    """Split the subject index 80/20, keeping the class balance in both halves.

    The split happens at subject level, never at time-step level: all 200 rows
    of a subject move together, so no part of a test subject is ever seen
    during training.

    Args:
        subject_metadata: Output of :func:`build_subject_metadata`.

    Returns:
        Tuple ``(train_metadata, test_metadata)``.
    """
    return train_test_split(
        subject_metadata,
        test_size=TEST_SIZE,
        stratify=subject_metadata["sii_binary"],
        random_state=SEEDS["train_test_split"],
    )


def build_split(dataset, split_metadata):
    """Extract the feature frames and labels for one side of the split.

    Args:
        dataset: List of per-subject DataFrames.
        split_metadata: Rows of the metadata table belonging to this split.

    Returns:
        Tuple ``(features, labels)`` where ``features`` is a list of
        DataFrames with the metadata columns removed and ``labels`` is a 1-D
        integer array of one label per subject.
    """
    features = []
    labels = []

    for index in split_metadata["original_index"]:
        subject = dataset[index]
        # Drop the target, the calendar metadata and the subject id — an id
        # left in the features would let a model memorise individual children.
        features.append(subject.drop(columns=COLUMNS_TO_DROP, errors="ignore"))
        # The label is constant per subject; mode() reads it robustly even if
        # a stray row ever disagreed.
        labels.append(subject["sii_binary"].mode()[0])

    return features, np.array(labels)


def describe_split(y_train, y_test):
    """Print the size and class balance of both halves of the split.

    Args:
        y_train: Training labels.
        y_test: Test labels.
    """
    print(f"Train X size (Subjects): {len(y_train)} | "
          f"Train y size (Labels): {len(y_train)}")
    print(f"Test X size (Subjects):  {len(y_test)} | "
          f"Test y size (Labels):  {len(y_test)}")
    print("-" * 50)

    for name, labels in (("Train", y_train), ("Test", y_test)):
        classes, counts = np.unique(labels, return_counts=True)
        print(f"{name} set class distribution:")
        for class_value, count in zip(classes, counts):
            print(f"  Class {class_value}: {count} elements "
                  f"({count / len(labels) * 100:.2f}%)")


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def remove_outliers_hampel(subject_frames):
    """Blank out sensor spikes with a per-feature Hampel filter.

    The Hampel filter slides a rolling median across a signal and flags any
    point sitting more than `sigma` scaled median-absolute-deviations away from
    that median. Flagged points become NaN — they are *not* replaced here; the
    interpolation step that follows fills them.

    Window and sigma come from ``config.HAMPEL_CONFIGS``, which tunes them per
    channel; the binary `non-wear_flag` is skipped entirely because a rolling
    median over a 0/1 flag is meaningless.

    Args:
        subject_frames: List of per-subject feature DataFrames.

    Returns:
        Tuple ``(cleaned_frames, outlier_records)``. ``cleaned_frames`` mirrors
        the input with outliers set to NaN; ``outlier_records`` is a list of
        dicts describing every value removed.
    """
    cleaned_frames = []
    outlier_records = []

    for subject_index, original in enumerate(subject_frames):
        cleaned = original.copy()

        for column in original.columns:
            if column in HAMPEL_SKIP_COLUMNS:
                continue

            settings = HAMPEL_CONFIGS.get(column, HAMPEL_DEFAULT)
            hampel = HampelFilter(window_length=settings["window"],
                                  n_sigma=settings["sigma"])
            filtered = hampel.fit_transform(original[column])

            # A value was an outlier if it was present before and is NaN after.
            was_removed = original[column].notna() & filtered.isna()
            for row_position in np.where(was_removed)[0]:
                outlier_records.append({
                    "subject_index": subject_index,
                    "timestamp": original.index[row_position],
                    "feature_name": column,
                    "original_value": original[column].iloc[row_position],
                })

            cleaned[column] = filtered

        cleaned_frames.append(cleaned)

    return cleaned_frames, outlier_records


def save_outlier_log(outlier_records):
    """Write every removed value to CSV so the filter's effect is auditable.

    Args:
        outlier_records: The record list from :func:`remove_outliers_hampel`.
    """
    if not outlier_records:
        print("Processing complete, but no outliers crossed the "
              "per-feature thresholds.")
        return

    outliers = pd.DataFrame(outlier_records)
    outliers.to_csv(PATHS["outlier_log"], index=False)
    print(f"All {len(outliers)} filtered outliers are saved to: "
          f"'{PATHS['outlier_log']}'")


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute_missing(subject_frames):
    """Fill the NaN holes the Hampel filter left, one subject at a time.

    Linear interpolation reconnects the signal across each gap; the trailing
    forward/backward fill catches gaps at the very start or end of a series,
    where interpolation has no value on one side to work from.

    Args:
        subject_frames: List of per-subject DataFrames containing NaNs.

    Returns:
        A new list of DataFrames with no missing values.
    """
    imputed = []
    for subject in subject_frames:
        filled = subject.copy()
        filled = filled.interpolate(method="linear", limit_direction="both")
        filled = filled.ffill().bfill()
        imputed.append(filled)
    return imputed


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def fit_feature_scalers(train_frames):
    """Fit one StandardScaler per sensor channel on the training subjects.

    All training subjects are stacked end to end so each scaler learns a single
    population mean and standard deviation for its channel, rather than
    per-subject statistics. Z-scoring this way strips out baseline differences
    between children while preserving the shape of their activity peaks —
    exactly what the distance-based models (k-NN, DTW, shapelets) need.

    Args:
        train_frames: List of imputed training DataFrames.

    Returns:
        Dict mapping column name -> fitted StandardScaler.
    """
    stacked_train = pd.concat(train_frames, axis=0)

    scalers = {}
    for column in SCALED_FEATURES:
        scaler = StandardScaler()
        scaler.fit(stacked_train[[column]])
        scalers[column] = scaler
    return scalers


def normalize_subjects(subject_frames, scalers):
    """Apply the fitted scalers to a list of subjects.

    `non-wear_flag` is left alone: it is a 0/1 indicator, so it is copied
    through unchanged and keeps its original column position.

    Args:
        subject_frames: List of per-subject DataFrames.
        scalers: Dict of fitted scalers from :func:`fit_feature_scalers`.

    Returns:
        A new list of DataFrames with the sensor channels standardised.
    """
    normalized = []
    for subject in subject_frames:
        scaled = subject.copy()
        for column in SCALED_FEATURES:
            scaled[column] = scalers[column].transform(
                subject[[column]]).flatten()
        normalized.append(scaled)
    return normalized


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_prepared_arrays(train_frames, test_frames, y_train, y_test):
    """Stack the per-subject frames into 3-D arrays and write them to disk.

    Args:
        train_frames: Normalised training DataFrames.
        test_frames: Normalised test DataFrames.
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        Tuple ``(X_train_array, X_test_array)`` of shape
        ``(n_subjects, 200, 8)``.
    """
    X_train_array = np.stack([frame.values for frame in train_frames])
    X_test_array = np.stack([frame.values for frame in test_frames])

    np.save(PATHS["X_train"], X_train_array)
    np.save(PATHS["X_test"], X_test_array)
    np.save(PATHS["y_train"], y_train)
    np.save(PATHS["y_test"], y_test)

    print("Successfully saved preprocessed datasets!")
    print(f"train_features.npy shape: {X_train_array.shape}")
    print(f"test_features.npy shape:  {X_test_array.shape}")
    print(f"train_labels.npy shape: {y_train.shape}")
    print(f"test_labels.npy shape:  {y_test.shape}")

    return X_train_array, X_test_array
