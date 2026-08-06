# ============================================================================================
# PULSAR: Advancing Interval-Based Time Series Classification to State-of-the-Art Performance
# Authors: Nestor Cabello, Lars Kulik
# Reference: IEEE International Conference on Data Mining (ICDM), 2025
# ============================================================================================

import os
import numpy as np
from aeon.datasets import load_from_ts_file
from aggregation_functions import AggregationFunctions
from pooling_operators import PoolingOperators
from numpy.lib.stride_tricks import as_strided
import random
from time import time
from itertools import combinations


# =========================
# Loading datasets
# =========================

def get_dataset_names_142(datasets_path):
    """
    List UCR dataset names found under a root path.

    Parameters
    ----------
    datasets_path : str
        Path containing one subdirectory per dataset.

    Returns
    -------
    list of str
        Sorted dataset names (folder names).
    """
    dataset_names = []
    for dataset_name in os.listdir(datasets_path):
        if os.path.isdir(os.path.join(datasets_path, dataset_name)):
            dataset_names.append(dataset_name)
    return sorted(dataset_names)


def get_train_test_files(datasets_folds_path, dataset_name, resample_id):
    """
    Build paths to TRAIN/TEST .ts files for a given resample.

    Parameters
    ----------
    datasets_folds_path : str
        Root path containing resampled folds.
    dataset_name : str
        Dataset name (folder and file stem).
    resample_id : int or str
        Resample identifier (e.g., 0).

    Returns
    -------
    (str, str)
        Tuple (train_file, test_file).
    """
    train_file = os.path.join(
        datasets_folds_path, dataset_name, dataset_name + str(resample_id) + "_TRAIN.ts"
    )
    test_file = os.path.join(
        datasets_folds_path, dataset_name, dataset_name + str(resample_id) + "_TEST.ts"
    )
    return train_file, test_file


def get_default_train_test_files(datasets_path, dataset_name):
    """
    Build default (non-resampled) TRAIN/TEST file paths.

    Parameters
    ----------
    datasets_path : str
        Root path with one folder per dataset.
    dataset_name : str
        Dataset name.

    Returns
    -------
    (str, str)
        Tuple (train_file, test_file).
    """
    train_file = os.path.join(datasets_path, dataset_name, dataset_name + "_TRAIN.ts")
    test_file = os.path.join(datasets_path, dataset_name, dataset_name + "_TEST.ts")
    return train_file, test_file


def get_default_train_test_sets(datasets_path, dataset_name):
    """
    Load default TRAIN/TEST sets for a dataset.

    Parameters
    ----------
    datasets_path : str
        Root path with datasets.
    dataset_name : str
        Dataset name.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        X_train, y_train, X_test, y_test loaded via `aeon`.
    """
    train_file, test_file = get_default_train_test_files(datasets_path, dataset_name)
    X_train, y_train = load_from_ts_file(train_file)
    X_test, y_test = load_from_ts_file(test_file)
    return X_train, y_train, X_test, y_test


def get_resample_indices(datasets_resample_indices_path, dataset_name, train_or_test, resample_id):
    """
    Load resample indices for TRAIN or TEST.

    Parameters
    ----------
    datasets_resample_indices_path : str
        Root path containing per-dataset resample index files.
    dataset_name : str
        Dataset name.
    train_or_test : {'TRAIN', 'TEST'}
        Which split to load.
    resample_id : int or str
        Resample identifier.

    Returns
    -------
    np.ndarray (int32)
        1D array of indices.
    """
    resample_indices_file = os.path.join(
        datasets_resample_indices_path,
        dataset_name,
        "resample" + str(resample_id) + "Indices" + "_" + train_or_test + ".txt",
    )
    print(resample_indices_file)
    return np.loadtxt(resample_indices_file, dtype=int)


# =========================
# Interval & pooling helpers
# =========================

def generate_fixed_intervals(input_length, list_interval_lengths, max_dilation):
    """
    Enumerate (length, dilation) pairs under a power-of-two dilation grid.

    For each base interval length L in `list_interval_lengths` (L <= input_length),
    include dilations d = 2^e up to both the data-implied maximum and the user cap.

    Parameters
    ----------
    input_length : int
        Length of the (transformed) series.
    list_interval_lengths : list[int]
        Candidate base lengths (e.g., [7, 9, 11]).
    max_dilation : int or None
        Optional upper bound on dilation; if None, only the data-implied maximum is used.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Two int32 arrays of equal length:
        - interval_lengths[i]
        - interval_dilations[i]
        such that each pair (length, dilation) is valid on the series.
    """
    list_interval_lengths = np.array(list_interval_lengths, dtype=np.int32)

    interval_lengths = []
    interval_dilations = []

    for interval_length in list_interval_lengths:
        if interval_length > input_length:
            continue

        # Max dilation so the last index stays within bounds.
        max_possible_dilation = (input_length - 1) // (interval_length - 1)
        if max_possible_dilation < 1:
            max_possible_dilation = 1

        max_possible_exponent = np.log2(max_possible_dilation)

        if max_dilation:
            max_dilation_exponent = np.log2(max_dilation)
            max_exponent = min(max_possible_exponent, max_dilation_exponent)
        else:
            max_exponent = max_possible_exponent

        for exp in range(int(np.floor(max_exponent)) + 1):
            dilation = 2 ** exp
            interval_lengths.append(interval_length)
            interval_dilations.append(dilation)

    return np.array(interval_lengths, dtype=np.int32), np.array(interval_dilations, dtype=np.int32)


def get_partitions(num_segments, num_parts):
    """
    Split a sequence of `num_segments` into `num_parts` nearly equal contiguous blocks.

    Parameters
    ----------
    num_segments : int
        Total number of segments.
    num_parts : int
        Desired number of partitions.

    Returns
    -------
    list[tuple[int, int]]
        List of (start_idx, end_idx) half-open intervals that cover [0, num_segments).
    """
    partition_size = num_segments // num_parts
    remainder = num_segments % num_parts
    partitions = []
    idx = 0
    for i in range(num_parts):
        start_idx = idx
        end_idx = idx + partition_size + (1 if i < remainder else 0)
        partitions.append((start_idx, end_idx))
        idx = end_idx
    return partitions


def extract_strided_segments(X, length, dilation):
    """
    Extract all sliding segments (per example) using a strided view.

    Parameters
    ----------
    X : np.ndarray, shape (n_examples, input_length)
        Batch of univariate series.
    length : int
        Segment length (number of points).
    dilation : int
        Step between consecutive points inside a segment.

    Returns
    -------
    np.ndarray, shape (n_examples, n_segments, length)
        Read-only strided view. `n_segments = input_length - (length - 1) * dilation`.
    """
    num_examples, input_length = X.shape
    step = dilation
    max_end = input_length - (length - 1) * step

    strided = as_strided(
        X,
        shape=(num_examples, max_end, length),
        strides=(X.strides[0], X.strides[1], step * X.strides[1]),
        writeable=False,
    )
    return strided


def compute_global_and_local_features_train(
    X,
    interval_lengths,
    interval_dilations,
    local_statistics,
    global_statistics,
    depth_local_features,
    initial_local_pooled_feature_index,
    num_random_selected_pooling_operators_per_interval,
    ts_representation,
):
    """
    Compute pooled features for TRAIN and record metadata needed for TEST-time filtering.

    Pipeline per (length, dilation):
      1) Extract all segments (strided).
      2) Compute all requested local stats per segment (vectorized).
      3) Build hierarchical partitions up to `depth_local_features`.
      4) For level==0 (global), include all global operators.
         For higher levels, randomly subsample operators per partition.
      5) Pool over partitions for each (local_stat, operator); record metadata.

    Parameters
    ----------
    X : np.ndarray, shape (n_examples, input_length)
        Transformed training series (one representation).
    interval_lengths : np.ndarray[int]
        Interval lengths for this representation.
    interval_dilations : np.ndarray[int]
        Interval dilations for this representation.
    local_statistics : list[str]
        Local stats names passed to `AggregationFunctions.transform_multiple`.
    global_statistics : list[str]
        Pooling operator names passed to `PoolingOperators.transform_individual`.
    depth_local_features : int
        Maximum number of hierarchy levels to generate (starting from level 0 for the global partition;
        each subsequent level splits partitions in two, e.g., level 1 -> 2 parts, level 2 -> 4 parts, etc.).
    initial_local_pooled_feature_index : int
        Starting index for numbering new local pooled features (updated as features are added).
    num_random_selected_pooling_operators_per_interval : int
        Number of operators to sample per partition at levels > 0.
    ts_representation : str
        Identifier of the current representation (for metadata).

    Returns
    -------
    (np.ndarray, np.ndarray, list, list, list, list, list)
        final_local_features : (n_examples, n_local) or empty array
        final_global_features : (n_examples, n_global) or empty array
        selected_operators_per_interval : list
            For each interval j, a list per partition of chosen operators.
        total_list_indices_count : list
            Metadata rows for local features (non-global, level>0).
        total_list_indices_count_global : list
            Metadata rows for global features (level==0).
        lst_partitions : list
            For each interval j: list of (start,end) partitions across levels.
        lst_levels : list
            For each interval j: list of level indices aligned with lst_partitions.

    Notes
    -----
    - Metadata rows are:
      [feature_idx, ts_representation, interval_length, interval_dilation,
       level, partition_number, start_idx, end_idx, global_pooling_operator, local_stat]
    - feature_idx==-1 is used for global features; local features increment from
      `initial_local_pooled_feature_index`.
    - These outputs help make TEST-time feature computation more efficient:
      after Fisher score-based feature selection is applied on the TRAIN features
      (outside this function, in `fit_transform`), the computation of TEST features
      not selected as relevant during training can be skipped.
    """
    num_examples, input_length = X.shape
    num_intervals = len(interval_lengths)
    pooled_features = []
    global_features = []
    selected_operators_per_interval = []

    total_list_indices_count = []
    total_list_indices_count_global = []

    num_local_stats = len(local_statistics)

    lst_partitions = []
    lst_levels = []

    for j in range(num_intervals):
        interval_length = interval_lengths[j]
        interval_dilation = interval_dilations[j]

        # Segments: (n_examples, n_segments, seg_len)
        segments = extract_strided_segments(X, interval_length, interval_dilation)

        n_ex, n_seg, seg_len = segments.shape
        reshaped = segments.reshape(-1, seg_len)

        # Compute all requested local stats once, then reshape to (n_examples, n_stats, n_segments)
        segment_stats = AggregationFunctions.transform_multiple(reshaped, local_statistics)
        segment_stats = segment_stats.reshape(n_ex, n_seg, num_local_stats).transpose(0, 2, 1)
        segment_stats = np.ascontiguousarray(segment_stats)

        num_segments = segment_stats.shape[2]

        # Hierarchy setup: num_parts = 1, 2, 4, ... up to depth and feasible by n_segments
        # Each 'num_parts' produces that many contiguous partitions of the segment index range.
        max_levels = depth_local_features
        max_possible_level = int(np.log2(num_segments)) + 1
        max_level = min(max_possible_level, max_levels)
        num_parts_list = [2 ** i for i in range(0, max_level)]

        lst_partitions.append([])
        lst_levels.append([])
        selected_operators_for_current_interval = []

        for num_parts in num_parts_list:
            partitions = get_partitions(num_segments, num_parts)
            lst_partitions[-1].extend(partitions) # Persist partitions for this interval j; needed later for TEST to mirror TRAIN.

            # Convert num_parts (1, 2, 4, ...) to level index (0, 1, 2, ...)
            level = int(np.log2(num_parts))
            lst_levels[-1].extend(level * np.ones(len(partitions), dtype=np.int32)) # record which hierarchical level each partition belongs to

            if num_parts > num_segments:
                continue # If we ask for more parts than segments, pooling is meaningless.

            # Operator selection policy:
            # - Level 0 (global, num_parts==1): use ALL operators (deterministic).
            # - Higher levels: randomly subsample to control feature explosion and compute cost.
            selected_operators_list = []
            for _ in partitions:
                if num_parts == 1:  # global level
                    selected_operators = global_statistics
                else:
                    selected_operators = random.sample(
                        global_statistics,
                        min(num_random_selected_pooling_operators_per_interval, len(global_statistics)),
                    )
                selected_operators_list.append(selected_operators)

            # Keep for this interval j; aligned by partition index.
            selected_operators_for_current_interval.extend(selected_operators_list)

        # Append per-interval operator selections once (shape: [partitions across all levels])
        selected_operators_per_interval.append(selected_operators_for_current_interval)

        # Pool for each (local_stat, operator, partition)
        for local_stat_idx in range(num_local_stats):
            single_stat_segment_stats = segment_stats[:, local_stat_idx, :]

            for global_stat in global_statistics:
                for partition_idx, partition in enumerate(lst_partitions[j]):
                    selected_operators = selected_operators_per_interval[j][partition_idx]
                    if global_stat not in selected_operators:
                        continue # This operator wasn't chosen for this partition at TRAIN

                    start_idx, end_idx = partition
                    if end_idx - start_idx == 1:
                        continue  # nothing to pool

                    segment = single_stat_segment_stats[:, start_idx:end_idx]
                    pooled_features_ = PoolingOperators.transform_individual(segment, global_stat)

                    level = lst_levels[j][partition_idx]
                    partition_number = partition_idx

                    # Metadata:
                    # - feature_idx==-1 tags global features (level 0).
                    # - local pooled features get a running column index (for reproducible mapping).
                    row = [
                        (-1 if level == 0 else initial_local_pooled_feature_index),
                        ts_representation,
                        interval_length,
                        interval_dilation,
                        level,
                        partition_number,
                        start_idx,
                        end_idx,
                        global_stat,
                        local_statistics[local_stat_idx],
                    ]

                    if level == 0:
                        total_list_indices_count_global.append([*row])
                        global_features.append(pooled_features_)
                    else:
                        total_list_indices_count.append([*row])
                        initial_local_pooled_feature_index += 1
                        pooled_features.append(pooled_features_)

    final_local_features = np.column_stack(pooled_features) if pooled_features else np.array([])
    final_global_features = np.column_stack(global_features) if global_features else np.array([])

    return (
        final_local_features,
        final_global_features,
        selected_operators_per_interval,
        total_list_indices_count,
        total_list_indices_count_global,
        lst_partitions,
        lst_levels,
    )


def compute_global_and_local_features_test(
    X,
    interval_lengths,
    interval_dilations,
    local_statistics,
    global_statistics,
    selected_operators_per_interval,
    ts_representation,
    lst_partitions,
    lst_levels,
    relevant_features_dictionary,
):
    """
    Compute pooled TEST features using TRAIN-time operator choices and feature filter.

    Only features that appeared in TRAIN (as keys in `relevant_features_dictionary`)
    are produced; others are returned as zero vectors to preserve column alignment.

    Parameters
    ----------
    X : np.ndarray, shape (n_examples, input_length)
        Transformed test series (one representation).
    interval_lengths : np.ndarray[int]
        Interval lengths used in TRAIN for this representation.
    interval_dilations : np.ndarray[int]
        Interval dilations used in TRAIN for this representation.
    local_statistics : list[str]
        Local stats (same order as TRAIN).
    global_statistics : list[str]
        Pooling operators (same order as TRAIN).
    selected_operators_per_interval : list
        TRAIN-time per-partition operator selections.
    ts_representation : str
        Representation identifier (for dictionary keys).
    lst_partitions : list
        Per-interval partitions (aligned with TRAIN).
    lst_levels : list
        Per-interval levels (aligned with TRAIN partitions).
    relevant_features_dictionary : dict
        Keys of the form
        "{rep},{L},{d},{local_stat},{global_stat},{start},{end}" for features kept at TRAIN.

    Returns
    -------
    (np.ndarray, np.ndarray)
        final_local_features : (n_examples, n_local) or empty array
        final_global_features : (n_examples, n_global) or empty array
    """
    num_examples, input_length = X.shape
    num_intervals = len(interval_lengths)
    pooled_features = []
    global_features = []

    num_local_stats = len(local_statistics)

    for j in range(num_intervals):
        interval_length = interval_lengths[j]
        interval_dilation = interval_dilations[j]

        # Mirror TRAIN: same (length, dilation) -> same segments
        segments = extract_strided_segments(X, interval_length, interval_dilation)

        n_ex, n_seg, seg_len = segments.shape
        reshaped = segments.reshape(-1, seg_len)

        # Same local stats ordering as TRAIN
        combined = AggregationFunctions.transform_multiple(reshaped, local_statistics)
        segment_stats = combined.reshape(n_ex, n_seg, num_local_stats).transpose(0, 2, 1)
        segment_stats = np.ascontiguousarray(segment_stats)

        for local_stat_idx in range(num_local_stats):
            single_stat_segment_stats = segment_stats[:, local_stat_idx, :]

            for global_stat in global_statistics:
                for partition_idx, partition in enumerate(lst_partitions[j]):
                    level = lst_levels[j][partition_idx]

                    # Respect TRAIN-time operator choices; if not selected, skip to keep alignment.
                    selected_ops = selected_operators_per_interval[j][partition_idx]
                    if global_stat not in selected_ops:
                        continue

                    start_idx, end_idx = partition
                    if end_idx - start_idx == 1:
                        continue # nothing to pool

                    key = f"{ts_representation},{interval_length},{interval_dilation},{local_statistics[local_stat_idx]},{global_stat},{start_idx},{end_idx}"
                    segment = single_stat_segment_stats[:, start_idx:end_idx]

                    if level == 0:
                        # Global features are always computed (they are unfiltered at TRAIN).
                        pooled_features_ = PoolingOperators.transform_individual(segment, global_stat)
                        global_features.append(pooled_features_)
                    else:
                        # Local features: only compute if TRAIN kept this key.
                        # Otherwise emit zeros to preserve column alignment across folds/splits.
                        if key in relevant_features_dictionary:
                            pooled_features_ = PoolingOperators.transform_individual(segment, global_stat)
                        else:
                            pooled_features_ = np.zeros((num_examples,), dtype=np.float32)
                        pooled_features.append(pooled_features_)

    final_local_features = np.column_stack(pooled_features) if pooled_features else np.array([])
    final_global_features = np.column_stack(global_features) if global_features else np.array([])

    return final_local_features, final_global_features
