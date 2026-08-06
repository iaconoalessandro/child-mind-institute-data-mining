"""
Small shared helpers used across the pipeline.

Nothing here is specific to regression or classification: it is the plumbing
(platform quirks, reading Optuna result files back, type-casting) that the
other modules need.
"""

import ast
import glob
import os
import platform

import pandas as pd


def get_n_jobs(default=-1):
    """Return a safe worker count for ``joblib.Parallel``.

    On macOS, nesting joblib's process pool inside PyTorch/BLAS threads can
    hang or crash the kernel, so we fall back to a single worker there.

    Args:
        default: Worker count to use on non-macOS platforms.

    Returns:
        1 on macOS, otherwise ``default``.
    """
    if platform.system() == "Darwin":
        return 1
    return default


# ---------------------------------------------------------------------------
# Reading Optuna results back from disk
# ---------------------------------------------------------------------------

def get_best_params_from_csv(filename):
    """Read the winning hyperparameters out of an Optuna trials CSV.

    The tuning stages save each study's ``trials_dataframe()`` to CSV. The best
    trial is the completed one with the largest ``value`` — both tracks
    maximise their metric (QWK for regression, Macro F1 for classification).

    Args:
        filename: Path to an Optuna trials CSV.

    Returns:
        Dict of hyperparameter name -> value, with Optuna's ``params_`` prefix
        stripped. Returns an empty dict if the file is missing or has no
        completed trials.
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return {}

    trials = pd.read_csv(filename)
    if "value" not in trials.columns:
        return {}

    if "state" in trials.columns:
        trials = trials[trials["state"] == "COMPLETE"]
    if trials.empty:
        print(f"No complete trials found in {filename}")
        return {}

    best_trial = trials.loc[trials["value"].idxmax()]

    best_params = {}
    for column in trials.columns:
        if not column.startswith("params_") or pd.isnull(best_trial[column]):
            continue
        value = best_trial[column]
        # Categorical parameters round-trip through the CSV as strings; turn
        # "0.5" or "[32, 64]" back into real Python objects where possible and
        # leave genuine strings (e.g. "rbf") untouched.
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                pass
        best_params[column.replace("params_", "")] = value
    return best_params


def load_tuned_params(csv_path):
    """Load best params from a trials CSV and reshape them for a constructor.

    Two things need fixing up before the values can be handed to a model:

    1. ``pca_n_components`` describes the preprocessing pipeline, not the
       model, so it is pulled out and returned separately.
    2. The neural network is tuned as ``n_layers`` plus one ``n_units_lN`` per
       layer; the estimator instead wants a single ``hidden_layers`` list.

    Args:
        csv_path: Path to an Optuna trials CSV.

    Returns:
        Tuple ``(params, pca_n_components)``. ``pca_n_components`` is None when
        the study did not tune PCA.
    """
    params = get_best_params_from_csv(csv_path)
    if not params:
        return {}, None

    pca_n = None
    if "pca_n_components" in params:
        pca_n = int(params.pop("pca_n_components"))

    # Collapse n_layers + n_units_l1..N into one hidden_layers list.
    if "n_layers" in params:
        n_layers = int(params.pop("n_layers"))
        hidden_layers = []
        for layer in range(1, n_layers + 1):
            width = params.pop(f"n_units_l{layer}", None)
            if width is not None and not pd.isna(width):
                hidden_layers.append(int(width))
        params["hidden_layers"] = hidden_layers

    # Drop any leftover width entries from trials that used more layers.
    for key in [k for k in params if k.startswith("n_units")]:
        del params[key]

    # These belong to the preprocessing / optimizer setup, not the estimator.
    for key in ["imputer_choice", "optimizer_secondary"]:
        params.pop(key, None)

    return params, pca_n


def load_tuned_params_by_prefix(study_prefix, results_dir):
    """Find a trials CSV by filename fragment and load its best params.

    Convenience wrapper around :func:`load_tuned_params` for when you know the
    model nickname (``"lgb"``, ``"xgb"``, ...) but not the exact filename. If
    several files match, the most recently created one wins.

    Args:
        study_prefix: Fragment that appears in the filename, e.g. ``"lgb"``.
        results_dir: Directory holding the trial CSVs.

    Returns:
        Tuple ``(params, pca_n_components)``, or ``({}, None)`` if no match.
    """
    matches = glob.glob(os.path.join(results_dir, f"*{study_prefix}*.csv"))
    if not matches:
        matches = glob.glob(f"*{study_prefix}*.csv")
    if not matches:
        # Worth shouting about: with no tuned parameters the caller silently
        # builds the model with library defaults, which looks like it worked
        # but does not reproduce any reported result.
        print(f"WARNING: no trials CSV matching '*{study_prefix}*.csv' in "
              f"{results_dir!r} — the model will fall back to default "
              f"hyperparameters. Run the matching tuning stage first.")
        return {}, None
    return load_tuned_params(max(matches, key=os.path.getctime))


# Hyperparameters that must be Python ints. A CSV round-trip turns every
# numeric column into a float, and scikit-learn rejects e.g. n_estimators=150.0.
INTEGER_PARAMS = [
    "n_estimators", "max_depth", "num_leaves", "min_child_samples",
    "min_samples_split", "min_samples_leaf", "min_child_weight",
    "iterations", "depth", "degree", "batch_size", "epochs",
]


def cast_integer_params(params):
    """Return a copy of ``params`` with known integer-valued entries cast to int.

    Args:
        params: Hyperparameter dict, possibly holding floats read from a CSV.

    Returns:
        A new dict; the input is not modified.
    """
    cast = dict(params)
    for key in INTEGER_PARAMS:
        if key in cast:
            cast[key] = int(cast[key])
    return cast
