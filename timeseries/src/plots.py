"""
Diagnostic plots for the data-preparation stage.

Three figures, each answering one question about the preparation chain:

1. :func:`plot_outlier_treatment` — what did the Hampel filter actually remove,
   and what did interpolation put back?
2. :func:`plot_stl_decomposition` — how much of a subject's signal is trend,
   how much is daily routine, how much is noise?
3. :func:`plot_normalization_effect` — what does Z-scoring do to two subjects
   recorded at different baselines?

None of these feed the models; they exist so the preparation choices can be
inspected rather than taken on trust.
"""

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL

from src.config import PATHS

# STL needs a cycle length. 48 half-hour steps is one full day, so the
# "seasonal" component it extracts is the subject's daily activity routine.
STL_PERIOD = 48


def plot_outlier_treatment(original_frames, cleaned_frames, imputed_frames,
                           subject_index=0, feature="enmo"):
    """Show one signal before filtering, after filtering, and after imputation.

    Args:
        original_frames: Raw training DataFrames.
        cleaned_frames: The same subjects with outliers set to NaN.
        imputed_frames: The same subjects after interpolation.
        subject_index: Which subject to display.
        feature: Which sensor channel to display.
    """
    # Read back the audit log so the removed points can be marked on the plot.
    outliers = pd.read_csv(PATHS["outlier_log"])
    outliers["timestamp"] = outliers["timestamp"].astype(int)
    subject_outliers = outliers[
        (outliers["subject_index"] == subject_index)
        & (outliers["feature_name"] == feature)
    ]

    original_series = original_frames[subject_index][feature]
    cleaned_series = cleaned_frames[subject_index][feature]
    imputed_series = imputed_frames[subject_index][feature]

    fig, (ax_raw, ax_cleaned, ax_imputed) = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True)

    # Panel 1 — the raw signal, with every flagged point circled in red.
    ax_raw.plot(original_series.index, original_series.values,
                color="#7f8c8d", alpha=0.6, label="Original Raw Data",
                linewidth=1.5)
    if not subject_outliers.empty:
        ax_raw.scatter(subject_outliers["timestamp"],
                       subject_outliers["original_value"],
                       color="#e74c3c", edgecolor="black", s=45, zorder=5,
                       label=f"Flagged Outliers "
                             f"({len(subject_outliers)} points)")
    else:
        print(f"Note: no outliers recorded for Subject {subject_index}, "
              f"feature '{feature}'.")
    ax_raw.set_title(f"1. Raw Input Signal (Subject {subject_index} - "
                     f"{feature})", fontsize=12, fontweight="bold")
    ax_raw.set_ylabel("Value")

    # Panel 2 — the same signal with those points removed, leaving visible gaps.
    ax_cleaned.plot(cleaned_series.index, cleaned_series.values,
                    color="#d35400", label="Outliers Hollowed out (NaNs)",
                    linewidth=1.5)
    ax_cleaned.set_title("2. Cleaned Signal (Hampel Filter Output)",
                         fontsize=12, fontweight="bold")
    ax_cleaned.set_ylabel("Value")

    # Panel 3 — the gaps bridged by interpolation, ready for modelling.
    ax_imputed.plot(imputed_series.index, imputed_series.values,
                    color="#27ae60", label="Linearly Interpolated Signal",
                    linewidth=1.75)
    if not subject_outliers.empty:
        ax_imputed.scatter(
            subject_outliers["timestamp"],
            imputed_series.loc[subject_outliers["timestamp"]].values,
            marker="o", facecolors="none", edgecolors="#27ae60", s=60,
            zorder=5, label="Interpolated Patch Points")
    ax_imputed.set_title("3. Final Imputed Signal (Ready for Modeling / ML)",
                         fontsize=12, fontweight="bold")
    ax_imputed.set_xlabel("Sequence Timestamp Index (0 - 200)")
    ax_imputed.set_ylabel("Value")

    for axis in (ax_raw, ax_cleaned, ax_imputed):
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_stl_decomposition(imputed_frames, subject_index=0, feature="enmo"):
    """Split one subject's signal into trend, daily routine and noise.

    STL (Seasonal-Trend decomposition using Loess) separates a series into
    three additive parts. ``robust=True`` down-weights extreme points so a
    residual spike cannot drag the trend line with it.

    All four panels share one Y-scale, which makes the relative size of each
    component immediately readable: a tall seasonal band and a flat trend means
    the subject's behaviour is dominated by their daily cycle.

    Args:
        imputed_frames: Imputed training DataFrames.
        subject_index: Which subject to decompose.
        feature: Which sensor channel to decompose.
    """
    sensor_series = imputed_frames[subject_index][feature]
    decomposition = STL(sensor_series, period=STL_PERIOD, robust=True).fit()

    components = {
        "1. Level (Observed Clean Data)": decomposition.observed,
        "2. Long-Term Trend": decomposition.trend,
        "3. Seasonality (Cyclical Routine)": decomposition.seasonal,
        "4. Noise (Unexplained Residuals)": decomposition.resid,
    }

    # One shared Y-range across all four panels, with a 5% margin so the lines
    # are not clipped by the axes.
    global_min = min(series.min() for series in components.values())
    global_max = max(series.max() for series in components.values())
    margin = (global_max - global_min) * 0.05

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 11), sharex=True)
    colors = ["#2c3e50", "#2980b9", "#e67e22", "#95a5a6"]

    for axis, (title, series), color in zip(axes, components.items(), colors):
        axis.plot(series.index, series.values, color=color, linewidth=1.75,
                  label=feature.upper())
        axis.set_title(title, fontsize=12, fontweight="bold", loc="left")
        axis.set_ylabel("Sensor Amplitude")
        axis.set_ylim(global_min - margin, global_max + margin)
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel(
        f"Sequence Timestamp Index (Subject #{subject_index})",
        fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_normalization_effect(imputed_frames, normalized_frames,
                              first_subject=0, second_subject=3,
                              feature="enmo", plot_window=500):
    """Compare two subjects before and after Z-score normalisation.

    The point of the figure: raw traces sit at different baselines because
    devices and bodies differ, which makes a distance-based model treat two
    identically-shaped signals as far apart. After standardisation both sit
    around zero and only their shapes differ.

    Args:
        imputed_frames: Training DataFrames before normalisation.
        normalized_frames: The same subjects after normalisation.
        first_subject: Index of the first subject to plot.
        second_subject: Index of the second subject to plot.
        feature: Which sensor channel to plot.
        plot_window: How many leading time steps to show.
    """
    fig, (ax_raw, ax_normalized) = plt.subplots(1, 2, figsize=(15, 5),
                                                sharex=True)

    # Left — raw units, where the two subjects sit at different baselines.
    for subject_index, color in ((first_subject, "#1f77b4"),
                                 (second_subject, "#ff7f0e")):
        series = imputed_frames[subject_index][feature]
        ax_raw.plot(series.iloc[:plot_window],
                    label=f"Subject {subject_index} (Raw)", color=color,
                    linewidth=2)
    ax_raw.set_title(f"Before Standardization: Raw {feature}", fontsize=13,
                     fontweight="bold", pad=12)
    ax_raw.set_ylabel("Raw Sensor Units", fontsize=11)

    # Right — the same two subjects in standard deviations from the population
    # mean, so their shapes can be compared directly.
    for subject_index, color in ((first_subject, "#1f77b4"),
                                 (second_subject, "#ff7f0e")):
        series = normalized_frames[subject_index][feature]
        ax_normalized.plot(series.iloc[:plot_window],
                           label=f"Subject {subject_index} (Normalized)",
                           color=color, linewidth=2)
    ax_normalized.set_title(
        f"After Standardization: Standardized {feature}", fontsize=13,
        fontweight="bold", pad=12)
    ax_normalized.set_ylabel(
        r"Standard Deviations from Population Mean ($\sigma$)", fontsize=11)
    ax_normalized.axhline(0, color="black", linestyle=":", alpha=0.3,
                          label="Population Mean")

    for axis in (ax_raw, ax_normalized):
        axis.set_xlabel("Time Steps / Timestamps", fontsize=11)
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    plt.show()
