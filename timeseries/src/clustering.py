"""
STEP 3 — Approximation, clustering and matrix profile.

Three linked questions, in order:

1. **Approximation** — can 200 time steps be compressed to 40 without losing
   the shape? (PAA)
2. **Clustering** — do the compressed signals fall into recognisable groups of
   daily routine? (DTW k-means and hierarchical clustering)
3. **Matrix profile** — inside each group's average shape, which subsequences
   repeat (motifs) and which stand alone (discords)?

This stage is unsupervised: the labels are never used.
"""

import numpy as np
import stumpy
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from tslearn.piecewise import PiecewiseAggregateApproximation

import matplotlib.pyplot as plt

from src.config import (DTW_N_JOBS, K_SEARCH_RANGE, MATRIX_PROFILE_MAX_MOTIFS,
                        MATRIX_PROFILE_TOP_DISCORDS, MATRIX_PROFILE_WINDOW,
                        N_CLUSTERS, PAA_N_SEGMENTS, SAKOE_CHIBA_RADIUS, SEEDS,
                        TSNE_MAX_ITER, TSNE_PERPLEXITY)

# One colour per cluster, used by every plot in this module.
CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2"]


# ---------------------------------------------------------------------------
# Piecewise Aggregate Approximation
# ---------------------------------------------------------------------------

def compute_paa(X):
    """Compress each series from 200 time steps to 40 segment averages.

    PAA cuts a series into equal-width segments and replaces each with its
    mean. It is the cheapest useful time-series approximation: it removes
    high-frequency noise, cuts the length by 5x, and leaves the overall shape
    intact — which is what makes the (expensive) DTW step below affordable.

    Args:
        X: Array of shape ``(n_subjects, n_timesteps, n_features)``.

    Returns:
        Tuple ``(paa_3d, paa_flat)`` of shapes
        ``(n_subjects, 40, n_features)`` and ``(n_subjects, 40 * n_features)``.
    """
    paa = PiecewiseAggregateApproximation(n_segments=PAA_N_SEGMENTS)
    paa_3d = paa.fit_transform(X)

    # Flatten so the plain (Euclidean) tools below — KMeans, PCA — can work on
    # one row per subject.
    paa_flat = paa_3d.reshape(X.shape[0], -1)

    print("PAA Approximation complete!")
    print(f"3D PAA Shape: {paa_3d.shape}")
    print(f"Flattened PAA Shape for linear models: {paa_flat.shape}")
    return paa_3d, paa_flat


# ---------------------------------------------------------------------------
# How many clusters?
# ---------------------------------------------------------------------------

def search_cluster_count(paa_flat):
    """Score plain k-means for k = 2..10 with the elbow and silhouette methods.

    Two complementary views: inertia always falls as k grows, so you look for
    the "elbow" where it stops falling fast; the silhouette score rewards
    clusters that are tight and well separated, so you look for its peak.

    Args:
        paa_flat: Flattened PAA features, shape ``(n_subjects, n_features)``.

    Returns:
        Tuple ``(inertia_values, silhouette_scores)``, one entry per k.
    """
    inertia_values = []
    silhouette_scores = []

    print("Evaluating cluster numbers...")
    for k in K_SEARCH_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=SEEDS["kmeans"],
                        n_init="auto")
        labels = kmeans.fit_predict(paa_flat)

        inertia_values.append(kmeans.inertia_)
        score = silhouette_score(paa_flat, labels)
        silhouette_scores.append(score)
        print(f"k = {k} | Inertia: {kmeans.inertia_:.2f} | "
              f"Silhouette Score: {score:.4f}")

    return inertia_values, silhouette_scores


def plot_cluster_count_search(inertia_values, silhouette_scores):
    """Plot the elbow and silhouette curves side by side.

    Args:
        inertia_values: Inertia per k, from :func:`search_cluster_count`.
        silhouette_scores: Silhouette score per k.
    """
    k_values = list(K_SEARCH_RANGE)
    fig, (ax_elbow, ax_silhouette) = plt.subplots(1, 2, figsize=(16, 6))

    ax_elbow.plot(k_values, inertia_values, marker="o", color="tab:blue",
                  linewidth=2)
    ax_elbow.set_title("The Elbow Method (Minimize Inertia)", fontsize=13,
                       fontweight="bold", pad=10)
    ax_elbow.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)

    ax_silhouette.plot(k_values, silhouette_scores, marker="o",
                       color="tab:orange", linewidth=2)
    ax_silhouette.set_title("The Silhouette Method (Maximize Score)",
                            fontsize=13, fontweight="bold", pad=10)
    ax_silhouette.set_ylabel("Average Silhouette Coefficient", fontsize=11)

    for axis in (ax_elbow, ax_silhouette):
        axis.set_xlabel("Number of Clusters (k)", fontsize=11)
        axis.set_xticks(k_values)
        axis.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Finding the Optimal Number of Clusters (k)", fontsize=16,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# DTW-based clustering
# ---------------------------------------------------------------------------

def build_dtw_input(paa_flat):
    """Reshape the flattened PAA features into tslearn's 3-D input format.

    IMPORTANT — this is not the multivariate array from :func:`compute_paa`.
    The eight channels stay concatenated end to end, and the result is handed
    to DTW as a **single 320-step univariate series** per subject rather than
    as 40 steps across 8 channels. Every clustering number and centroid in this
    stage was produced this way, so it is kept exactly as-is; changing it would
    change every result below.

    Args:
        paa_flat: Flattened PAA features, shape ``(n_subjects, 320)``.

    Returns:
        Array of shape ``(n_subjects, 320, 1)``.
    """
    return paa_flat[:, :, np.newaxis]


def cluster_kmeans_dtw(dtw_input):
    """Cluster subjects with k-means using DTW as the distance measure.

    Dynamic Time Warping matches two series even when the same behaviour
    happens at slightly different times — the right notion of similarity for
    daily routines that are not perfectly synchronised. The Sakoe-Chiba band
    caps how far the alignment may drift, which keeps the cost tractable.
    Centroids are averaged in DTW space (DBA).

    Args:
        dtw_input: Array from :func:`build_dtw_input`.

    Returns:
        1-D array of cluster labels, one per subject.
    """
    print("Running DTW K-Means (Sakoe-Chiba constrained)...")
    kmeans_dtw = TimeSeriesKMeans(
        n_clusters=N_CLUSTERS,
        metric="dtw",
        metric_params={"global_constraint": "sakoe_chiba",
                       "sakoe_chiba_radius": SAKOE_CHIBA_RADIUS},
        max_iter=10,
        n_jobs=DTW_N_JOBS,
        random_state=SEEDS["kmeans"],
    )
    return kmeans_dtw.fit_predict(dtw_input)


def compute_dtw_distance_matrix(dtw_input):
    """Compute the full pairwise DTW distance matrix.

    Precomputing it once lets both the hierarchical clustering and the t-SNE
    projection below reuse the same distances instead of recomputing them.

    Args:
        dtw_input: Array from :func:`build_dtw_input`.

    Returns:
        Square array of shape ``(n_subjects, n_subjects)``.
    """
    print("Computing pairwise DTW distance matrix for Hierarchical "
          "Clustering...")
    return cdist_dtw(dtw_input, global_constraint="sakoe_chiba",
                     sakoe_chiba_radius=SAKOE_CHIBA_RADIUS, n_jobs=DTW_N_JOBS)


def cluster_hierarchical_dtw(distance_matrix):
    """Cluster the precomputed DTW distances with average-linkage agglomeration.

    Average linkage merges the two clusters whose members are closest on
    average — more robust to a single odd subject than single linkage, and less
    biased toward equal-sized clusters than Ward.

    Args:
        distance_matrix: Output of :func:`compute_dtw_distance_matrix`.

    Returns:
        1-D array of cluster labels, one per subject.
    """
    print("Running Hierarchical Clustering (Average Linkage on DTW "
          "distances)...")
    hierarchical = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="precomputed",   # required when passing distances, not features
        linkage="average",
    )
    return hierarchical.fit_predict(distance_matrix)


# ---------------------------------------------------------------------------
# Two-dimensional projections
# ---------------------------------------------------------------------------

def project_pca(paa_flat):
    """Project the PAA features onto their two principal components.

    A global linear view: fast, and it preserves the directions of largest
    variance, but it measures similarity with Euclidean distance rather than
    the DTW distance the clusters were actually built on.

    Args:
        paa_flat: Flattened PAA features.

    Returns:
        Array of shape ``(n_subjects, 2)``.
    """
    print("\nRunning PCA dimensionality reduction (Global Linear "
          "Projection)...")
    return PCA(n_components=2, random_state=SEEDS["pca"]).fit_transform(
        paa_flat)


def project_tsne(distance_matrix):
    """Project the DTW distance matrix into 2-D with t-SNE.

    Passing ``metric="precomputed"`` makes t-SNE lay out the actual DTW space
    rather than a Euclidean approximation of it, so the picture matches the
    distance the clusters were formed with. t-SNE preserves local neighbourhood
    structure, not global distances — read cluster shapes, not gaps.

    Args:
        distance_matrix: Output of :func:`compute_dtw_distance_matrix`.

    Returns:
        Array of shape ``(n_subjects, 2)``.
    """
    print("Running t-SNE dimensionality reduction using Precomputed DTW "
          "distances...")
    tsne = TSNE(n_components=2, metric="precomputed", init="random",
                perplexity=TSNE_PERPLEXITY, random_state=SEEDS["tsne"],
                max_iter=TSNE_MAX_ITER)
    return tsne.fit_transform(distance_matrix)


def plot_cluster_projections(pca_points, tsne_points, kmeans_labels,
                             hierarchical_labels):
    """Draw both clusterings in both projections as a 2x2 comparison grid.

    Reading the grid: agreement down a column means the two algorithms found
    the same structure; agreement across a row means that structure survives
    the change of projection.

    Args:
        pca_points: 2-D PCA coordinates.
        tsne_points: 2-D t-SNE coordinates.
        kmeans_labels: Labels from :func:`cluster_kmeans_dtw`.
        hierarchical_labels: Labels from :func:`cluster_hierarchical_dtw`.
    """
    def scatter_clusters(axis, points, labels, title, xlabel, ylabel):
        """Scatter one projection, colouring points by cluster membership."""
        for cluster in range(N_CLUSTERS):
            members = labels == cluster
            axis.scatter(points[members, 0], points[members, 1],
                         label=f"Cluster {cluster}", alpha=0.8,
                         edgecolors="w", s=70, color=CLUSTER_COLORS[cluster])
        axis.set_title(title, fontsize=12, fontweight="bold", pad=10)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(True, linestyle="--", alpha=0.5)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    scatter_clusters(axes[0, 0], pca_points, kmeans_labels,
                     "PCA Projection: K-Means (DTW) Clusters",
                     "Principal Component 1", "Principal Component 2")
    scatter_clusters(axes[0, 1], pca_points, hierarchical_labels,
                     "PCA Projection: Hierarchical (DTW) Clusters",
                     "Principal Component 1", "Principal Component 2")
    scatter_clusters(axes[1, 0], tsne_points, kmeans_labels,
                     "t-SNE (DTW Space) Projection: K-Means (DTW) Clusters",
                     "t-SNE Dimension 1", "t-SNE Dimension 2")
    scatter_clusters(axes[1, 1], tsne_points, hierarchical_labels,
                     "t-SNE (DTW Space) Projection: Hierarchical (DTW) "
                     "Clusters", "t-SNE Dimension 1", "t-SNE Dimension 2")

    plt.suptitle("DTW-Based Time Series Clustering Evaluation & Visualization",
                 fontsize=16, fontweight="bold", y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ---------------------------------------------------------------------------
# Matrix profile
# ---------------------------------------------------------------------------

def compute_cluster_centroids(dtw_input, labels):
    """Average the members of each cluster into one representative shape.

    Args:
        dtw_input: Array from :func:`build_dtw_input`.
        labels: Cluster label per subject.

    Returns:
        Array of shape ``(N_CLUSTERS, series_length)`` — one 1-D centroid per
        cluster, ready for the matrix profile.
    """
    print("Computing cluster centroids from saved labels...")
    centroids = np.array([
        np.mean(dtw_input[labels == cluster], axis=0)
        for cluster in range(N_CLUSTERS)
    ])

    # dtw_input carries a trailing singleton channel axis; drop it so each
    # centroid is a plain 1-D signal.
    centroids_2d = centroids[:, :, 0] if centroids.ndim == 3 else centroids
    print(f"Successfully computed centroids. Shape: {centroids_2d.shape}")
    return centroids_2d


def find_discords(matrix_profile, window, top_n):
    """Find the most isolated subsequences in a matrix profile.

    A discord is the subsequence whose nearest neighbour is furthest away — the
    least repeated, most anomalous stretch of the signal. After each pick, an
    exclusion zone of half a window is blanked out so the next rank cannot be a
    near-duplicate of the one just found.

    Args:
        matrix_profile: 1-D array of nearest-neighbour distances.
        window: Subsequence length used to build the profile.
        top_n: How many discords to return.

    Returns:
        List of ``(start_index, profile_distance)`` tuples, most anomalous
        first.
    """
    remaining = np.copy(matrix_profile)
    exclusion_zone = window // 2
    discords = []

    for _ in range(top_n):
        discord_index = np.argmax(remaining)
        discord_value = remaining[discord_index]

        # Stop early once only invalid or non-positive distances remain.
        if (np.isinf(discord_value) or np.isnan(discord_value)
                or discord_value <= 0):
            break

        discords.append((int(discord_index), float(discord_value)))
        start = max(0, discord_index - exclusion_zone)
        end = min(len(remaining), discord_index + exclusion_zone)
        remaining[start:end] = -np.inf

    return discords


def analyze_centroid_matrix_profiles(centroids, window=MATRIX_PROFILE_WINDOW):
    """Report the repeated motifs and the anomalies inside every centroid.

    The matrix profile stores, for every subsequence of length ``window``, the
    distance to its closest match elsewhere in the same series. Low values mark
    motifs (behaviour that recurs); high values mark discords (behaviour that
    happens once).

    Args:
        centroids: Array from :func:`compute_cluster_centroids`.
        window: Subsequence length ``m``.
    """
    for cluster_index, centroid in enumerate(centroids):
        print(f"\n{'=' * 50}")
        print(f"ANALYZING CLUSTER CENTROID {cluster_index}")
        print("=" * 50)

        # stumpy needs at least two non-overlapping windows to compare.
        if len(centroid) < 2 * window:
            print(f"Centroid length is too short for window size {window}. "
                  "Skipping...")
            continue

        matrix_profile = stumpy.stump(centroid, m=window)

        # --- Motifs: the most-repeated subsequences ---
        try:
            motif_distances, motif_indices = stumpy.motifs(
                centroid, matrix_profile[:, 0],
                max_motifs=MATRIX_PROFILE_MAX_MOTIFS, min_neighbors=1)

            print("Top Motifs Found:")
            motif_count = 0
            for rank, indices in enumerate(motif_indices):
                # stumpy pads short results with -1; drop those placeholders.
                valid = indices[indices != -1]
                if len(valid) == 0:
                    continue
                start_index = int(valid[0])
                distance = float(np.ravel(motif_distances[rank])[0])
                print(f"  Rank {rank + 1}: Subsequence starts at index "
                      f"{start_index} (Distance: {distance:.4f})")
                motif_count += 1
            if motif_count == 0:
                print("  No clear matching motifs found.")
        except Exception as error:
            print(f"  Could not extract motifs: {error}")

        # --- Discords: the least-repeated subsequences ---
        print(f"\nTop {MATRIX_PROFILE_TOP_DISCORDS} Anomalies (Discords) "
              "Found:")
        discords = find_discords(matrix_profile[:, 0], window,
                                 MATRIX_PROFILE_TOP_DISCORDS)
        for rank, (index, value) in enumerate(discords):
            print(f"  Rank {rank + 1}: Subsequence starts at index {index} "
                  f"(Profile Distance: {value:.4f})")
        if not discords:
            print("  No distinct anomalies found.")


def plot_centroid_diagnostics(centroids, cluster_index,
                              window=MATRIX_PROFILE_WINDOW):
    """Plot one centroid's top motif pair and top discord above its profile.

    Args:
        centroids: Array from :func:`compute_cluster_centroids`.
        cluster_index: Which centroid to inspect.
        window: Subsequence length ``m``.
    """
    centroid = centroids[cluster_index]
    matrix_profile = stumpy.stump(centroid, m=window)

    # The best motif is a pair: two places where the same shape occurs.
    try:
        _, motif_indices = stumpy.motifs(centroid, matrix_profile[:, 0],
                                         max_motifs=1, min_neighbors=1)
        first_occurrence = int(motif_indices[0][0])
        second_occurrence = int(motif_indices[0][1])
    except Exception:
        first_occurrence, second_occurrence = None, None

    # The discord is simply the profile's highest point.
    discord_index = int(np.argmax(matrix_profile[:, 0]))

    fig, (ax_signal, ax_profile) = plt.subplots(2, 1, figsize=(14, 8),
                                                sharex=True)

    # Top panel — the centroid, with the landmarks highlighted on top of it.
    ax_signal.plot(centroid, label="Centroid Shape", color="black", alpha=0.8,
                   linewidth=2)
    if first_occurrence is not None and first_occurrence != -1:
        ax_signal.plot(range(first_occurrence, first_occurrence + window),
                       centroid[first_occurrence:first_occurrence + window],
                       color="limegreen", linewidth=4,
                       label="Motif Occurrence 1")
    if second_occurrence is not None and second_occurrence != -1:
        ax_signal.plot(range(second_occurrence, second_occurrence + window),
                       centroid[second_occurrence:second_occurrence + window],
                       color="forestgreen", linewidth=4, linestyle="--",
                       label="Motif Occurrence 2 (Match)")
    ax_signal.plot(range(discord_index, discord_index + window),
                   centroid[discord_index:discord_index + window],
                   color="crimson", linewidth=4, label="Top Discord (Anomaly)")
    ax_signal.set_title(f"Cluster Centroid {cluster_index} - Structural "
                        "Landmarks", fontsize=14, fontweight="bold")
    ax_signal.set_ylabel("Amplitude")

    # Bottom panel — the profile itself, so the dips and peaks that produced
    # those landmarks are visible.
    ax_profile.plot(range(len(matrix_profile)), matrix_profile[:, 0],
                    color="royalblue", linewidth=1.5, label="Matrix Profile")
    if first_occurrence is not None and first_occurrence < len(matrix_profile):
        ax_profile.axvline(x=first_occurrence, color="forestgreen",
                           linestyle=":", alpha=0.8)
        ax_profile.scatter(first_occurrence,
                           matrix_profile[first_occurrence, 0],
                           color="forestgreen", s=100, zorder=5,
                           label="Motif Minimum")
    if discord_index < len(matrix_profile):
        ax_profile.axvline(x=discord_index, color="crimson", linestyle=":",
                           alpha=0.8)
        ax_profile.scatter(discord_index, matrix_profile[discord_index, 0],
                           color="crimson", s=100, zorder=5,
                           label="Discord Peak")
    ax_profile.set_title("Matrix Profile (Lower = Recurring Patterns | "
                         "Higher = Anomalies)", fontsize=12)
    ax_profile.set_xlabel("Time Step / Index")
    ax_profile.set_ylabel("Distance to Nearest Neighbor")

    for axis in (ax_signal, ax_profile):
        axis.legend(loc="upper right")
        axis.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
