"""
Small shared helpers used across the pipeline.

Nothing here is specific to one analysis stage: it is the plumbing (loading the
prepared arrays, reshaping them for the models, printing tidy headers) that the
other modules all need.
"""

import os
import sys

import numpy as np

from src.config import MODEL_CHANNELS, PATHS


def print_header(title, width=70):
    """Print a titled banner so long console runs stay readable.

    Args:
        title: Text to show between the two rule lines.
        width: Total character width of the rule lines.
    """
    print(f"\n{'=' * width}")
    print(title)
    print("=" * width)


def load_prepared_arrays():
    """Load the four arrays written by the preparation stage.

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)``. The two feature arrays
        have shape ``(n_subjects, 200, 8)`` — subjects, time steps, features —
        with the feature axis ordered as ``config.FEATURE_COLUMNS``.

    Raises:
        FileNotFoundError: If the preparation stage has not been run yet.
    """
    missing = [key for key in ("X_train", "X_test", "y_train", "y_test")
               if not os.path.exists(PATHS[key])]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared arrays: {missing}. "
            "Run `python main.py --stage prep` first."
        )

    return (np.load(PATHS["X_train"]), np.load(PATHS["X_test"]),
            np.load(PATHS["y_train"]), np.load(PATHS["y_test"]))


def select_model_channels(X):
    """Keep only the modelled channels and move them in front of time.

    Every classifier in this project works on enmo and anglez alone, and the
    time-series libraries expect ``(cases, channels, timesteps)`` whereas the
    prepared arrays are stored as ``(cases, timesteps, features)``.

    Args:
        X: Array of shape ``(n_subjects, n_timesteps, n_features)``.

    Returns:
        Array of shape ``(n_subjects, 2, n_timesteps)``.
    """
    return np.transpose(X[:, :, MODEL_CHANNELS], (0, 2, 1))


def add_pulsar_to_path():
    """Make the vendored PULSAR package importable.

    PULSAR ships as a folder of loose modules rather than an installable
    package, so its directory has to go on ``sys.path`` before
    ``from pulsar import PULSAR`` can work.
    """
    pulsar_path = os.path.abspath(PATHS["pulsar_dir"])
    if pulsar_path not in sys.path:
        sys.path.append(pulsar_path)


def z_normalize(values, epsilon):
    """Z-normalise a 1-D signal so only its shape matters, not its scale.

    Subtracting the mean removes any baseline offset and dividing by the
    standard deviation removes amplitude, which is what makes two subsequences
    with the same shape but different sizes compare as equal.

    Args:
        values: 1-D array to normalise.
        epsilon: Added to the standard deviation so a flat window cannot
            divide by zero.

    Returns:
        The normalised array, same shape as the input.
    """
    return (values - np.mean(values)) / (np.std(values) + epsilon)
