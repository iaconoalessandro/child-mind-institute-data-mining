"""
Consensus outlier detection.

Rather than trusting a single anomaly detector, we run ten of them — each
built on a different mathematical idea — and only call a row an outlier when
almost all of them agree. That makes the flag robust to any one method's
blind spots.

    statistical      HBOS, Grubbs (Mahalanobis on PCA)
    geometry/depth   Arning smoothing factor, ISODEPTH (Tukey depth)
    distance         k-nearest-neighbour
    density          LOF
    clustering       CBLOF
    high-dimensional ABOD (angle-based)
    ensemble         LODA, Isolation Forest

Note that LODA is not seeded by PyOD, so consensus flags can shift slightly
between runs. The saved `is_outlier` column in dataset/PostProcessed_*.csv is the
record of one specific run.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from src.config import OUTLIER_COMPARE_PERCENTILE, OUTLIER_MIN_VOTES, SEEDS

# Colours used when plotting the training set in 2D. Outliers are drawn last,
# fully opaque and in red; the majority class 0 is faded almost to the
# background so the rarer classes stay visible underneath it.
CLASS_COLOURS = {0: "#5F5F5F", 1: "#0008FF", 2: "#309D17", 3: "#FF00EA"}
OUTLIER_COLOUR = "#FF0000"
CLASS_ALPHAS = {0: 0.17, 1: 0.25, 2: 0.5, 3: 1.0}


# ---------------------------------------------------------------------------
# Score functions for the four methods PyOD does not provide
# ---------------------------------------------------------------------------

def grubbs_proxy_scores(X):
    """Multivariate Grubbs' test proxy: Mahalanobis distance in PCA space.

    Projects onto the components explaining 95% of variance, then measures each
    row's squared Mahalanobis distance from the centre. The covariance gets a
    tiny ridge (1e-6) on its diagonal before inversion so that near-singular
    directions do not blow the distance up.

    Args:
        X: 2D array, shape (n_samples, n_features), already scaled and NaN-free.

    Returns:
        1D array of scores; higher means more outlying.
    """
    projected = PCA(n_components=0.95, random_state=SEEDS["pca"]).fit_transform(X)
    centered = projected - np.mean(projected, axis=0)
    covariance = np.cov(projected, rowvar=False)
    inverse_covariance = np.linalg.pinv(
        covariance + np.eye(covariance.shape[0]) * 1e-6
    )
    # Kept as an explicit per-row loop: the batched equivalent differs in the
    # last few floating-point bits, and at ~7k rows this costs under 10 ms.
    return np.array([row @ inverse_covariance @ row.T for row in centered])


def arning_smoothing_factor_scores(X):
    """Arning et al. smoothing-factor proxy: squared distance from the mean.

    Args:
        X: 2D array, shape (n_samples, n_features).

    Returns:
        1D array of scores; higher means further from the centroid.
    """
    return np.sum((X - np.mean(X, axis=0)) ** 2, axis=1)


def isodepth_proxy_scores(X):
    """ISODEPTH (Tukey depth) proxy: distance from the median in a 2D PCA plane.

    Args:
        X: 2D array, shape (n_samples, n_features).

    Returns:
        1D array of scores; higher means less central, i.e. shallower depth.
    """
    projected = PCA(n_components=2, random_state=SEEDS["pca"]).fit_transform(X)
    return np.linalg.norm(projected - np.median(projected, axis=0), axis=1)


# ---------------------------------------------------------------------------
# Running all ten detectors
# ---------------------------------------------------------------------------

def prepare_matrix_for_detection(X):
    """Median-impute and standardise, because the detectors cannot take NaNs.

    Standardising matters for the distance- and density-based methods: without
    it, whichever feature has the largest raw units would dominate.

    Args:
        X: DataFrame or 2D array, possibly containing NaNs.

    Returns:
        2D numpy array with zero mean and unit variance per column.
    """
    filled = SimpleImputer(strategy="median").fit_transform(pd.DataFrame(X))
    return StandardScaler().fit_transform(filled)


def compute_outlier_scores(X):
    """Score every row with each of the ten detectors.

    Args:
        X: DataFrame or 2D array of features; NaNs are handled internally.

    Returns:
        dict mapping detector display name -> 1D score array. Insertion order
        is the canonical order the ten methods are listed in.
    """
    scaled = prepare_matrix_for_detection(X)
    seed = SEEDS["random_state"]

    return {
        "HBOS (Naive)": HBOS().fit(scaled).decision_scores_,
        "Grubbs (Stat)": grubbs_proxy_scores(scaled),
        "SF (Deviation)": arning_smoothing_factor_scores(scaled),
        "ISODEPTH (Depth)": isodepth_proxy_scores(scaled),
        "kNN (Distance)": KNN(method="largest").fit(scaled).decision_scores_,
        "LOF (Density)": LOF().fit(scaled).decision_scores_,
        "CBLOF (Cluster)": CBLOF(random_state=seed,
                                 check_estimator=False).fit(scaled).decision_scores_,
        "ABOD (High-Dim)": ABOD().fit(scaled).decision_scores_,
        "LODA (Ensemble)": LODA().fit(scaled).decision_scores_,
        "IForest (Model)": IForest(random_state=seed).fit(scaled).decision_scores_,
    }


def flag_top_scores(scores_by_method, percentile):
    """Turn raw scores into per-method 0/1 flags at a shared percentile cut.

    Each detector flags the rows scoring above its *own* ``percentile``-th
    score, so the methods stay comparable despite being on different scales.
    A percentile of 90 flags the top 10% of rows, 99 flags the top 1%.

    Args:
        scores_by_method: Output of :func:`compute_outlier_scores`.
        percentile: Cut-off percentile, e.g. 90 or 99.

    Returns:
        pd.DataFrame of 0/1 flags, one column per detector.
    """
    return pd.DataFrame({
        name: (scores > np.percentile(scores, percentile)).astype(int)
        for name, scores in scores_by_method.items()
    })


def detect_consensus_outliers(X, threshold_percentile, min_votes=OUTLIER_MIN_VOTES):
    """Flag rows that at least ``min_votes`` of the ten detectors call extreme.

    Args:
        X: DataFrame or 2D array of features; NaNs are handled internally.
        threshold_percentile: Per-method cut-off percentile (99 = top 1%).
        min_votes: How many of the ten detectors must agree.

    Returns:
        Boolean numpy array, True where the row is a consensus outlier.
    """
    flags = flag_top_scores(compute_outlier_scores(X), threshold_percentile)
    return flags.sum(axis=1).values >= min_votes


def plot_method_agreement(scores_by_method,
                          percentile=OUTLIER_COMPARE_PERCENTILE):
    """Show how much the ten detectors agree, and how many rows each flags.

    Left panel: correlation between the methods' flag vectors — a high value
    means two methods are finding the same structure, so they add little
    independent evidence to the vote. Right panel: how many rows each flags.

    Args:
        scores_by_method: Output of :func:`compute_outlier_scores`.
        percentile: Cut-off percentile for the flags, 90 by default (top 10%).
    """
    flags = flag_top_scores(scores_by_method, percentile)

    _, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(flags.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0])
    axes[0].set_title("Method Agreement (Correlation Matrix of Flags)",
                      fontsize=14)

    counts = flags.sum().sort_values(ascending=False)
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=axes[1])
    axes[1].set_title(f"Outliers Detected by Each Method "
                      f"(Top {100 - percentile}%)", fontsize=14)
    axes[1].set_ylabel("Number of Outliers")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()
    print("Interpretation: High correlation suggests methods detect similar "
          "structures.")


def plot_outlier_embeddings(X, labels, outlier_mask):
    """Project the data into six 2D spaces and colour the consensus outliers.

    If the flagged rows really are anomalous they should sit apart from the
    main cloud in at least some of these projections. Six different embeddings
    are used because each preserves a different property — global variance
    (PCA, TSVD), local neighbourhoods (t-SNE, UMAP, PaCMAP) or graph structure
    (Spectral) — and an artefact of one is unlikely to appear in all six.

    Args:
        X: Training features as a DataFrame; NaNs are k-NN imputed here.
        labels: Integer `sii` label per row.
        outlier_mask: Boolean array flagging the consensus outliers.
    """
    # Imported lazily: these two are only needed for this diagnostic plot.
    import pacmap
    import umap
    from sklearn.impute import KNNImputer

    labels = np.asarray(labels)
    outlier_mask = np.asarray(outlier_mask)

    imputed = KNNImputer(n_neighbors=29).fit_transform(X)
    scaled = StandardScaler().fit_transform(imputed)

    # Draw order: ordinary rows first, then the rare severe class, then
    # outliers on top, so nothing important is hidden under the majority class.
    colours = np.array([
        to_rgba(OUTLIER_COLOUR, alpha=1.0) if is_outlier
        else to_rgba(CLASS_COLOURS[label], alpha=CLASS_ALPHAS[label])
        for label, is_outlier in zip(labels, outlier_mask)
    ])
    draw_order = np.array([
        3 if is_outlier else (2 if label == 3 else 1)
        for label, is_outlier in zip(labels, outlier_mask)
    ])
    sorted_index = np.argsort(draw_order)
    scaled, colours = scaled[sorted_index], colours[sorted_index]

    embeddings = {
        "PCA": PCA(n_components=2),
        "TSVD": TruncatedSVD(n_components=2),
        "t-SNE": TSNE(n_components=2, init="pca", learning_rate="auto", n_jobs=4),
        "UMAP": umap.UMAP(n_components=2, n_jobs=4),
        "PaCMAP": pacmap.PaCMAP(n_components=2),
        "Spectral": SpectralEmbedding(n_components=2, n_jobs=4),
    }

    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    for axis, (name, method) in zip(axes.flatten(), embeddings.items()):
        try:
            projected = method.fit_transform(scaled)
            axis.scatter(projected[:, 0], projected[:, 1], c=colours, s=10,
                         edgecolors="none")
            axis.set_title(name, fontsize=14)
        except Exception:
            # One embedding failing should not lose the other five.
            axis.set_title(f"{name} (Failed)", fontsize=14)
        axis.set_xticks([])
        axis.set_yticks([])

    legend_entries = [
        Line2D([0], [0], marker="o", color="w", markersize=10,
               markerfacecolor=CLASS_COLOURS[label], label=f"SII={label}")
        for label in sorted(CLASS_COLOURS)
    ] + [
        Line2D([0], [0], marker="o", color="w", markersize=10,
               markerfacecolor=OUTLIER_COLOUR, label="Outlier")
    ]
    figure.legend(handles=legend_entries, loc="lower center", ncol=5,
                  bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.show()


def summarise_outliers(outlier_mask, labels):
    """Print how many outliers were found overall and within each `sii` class.

    Args:
        outlier_mask: Boolean array from :func:`detect_consensus_outliers`.
        labels: Matching array of integer `sii` labels.

    Returns:
        The ``outlier_mask`` it was given, for convenient chaining.
    """
    outlier_mask = np.asarray(outlier_mask)
    labels = np.asarray(labels)
    total = len(outlier_mask)
    flagged = int(outlier_mask.sum())

    print("=" * 50)
    print("CONSENSUS OUTLIER DETECTION (10 METHODS)")
    print("=" * 50)
    print(f"Total Outliers: {flagged} / {total} ({100 * flagged / total:.2f}%)")

    print("\nBreakdown by class (sii):")
    for label in sorted(np.unique(labels)):
        in_class = labels == label
        class_outliers = int(outlier_mask[in_class].sum())
        print(f" - Class {label}: {class_outliers} outliers "
              f"({100 * class_outliers / in_class.sum():.2f}% of class, "
              f"{100 * class_outliers / total:.2f}% of total)")

    return outlier_mask
