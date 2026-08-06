"""
Single entry point for the whole pipeline.

Run a stage by name:

    python main.py --stage prep        # build the .npy arrays from the pickle
    python main.py --stage cluster     # PAA, DTW clustering, matrix profile
    python main.py --stage shapelets   # shapelet search and rule extraction
    python main.py --stage classify    # train and evaluate the six classifiers
    python main.py --stage patterns    # sequential pattern mining
    python main.py --stage all         # everything, in order

Only ``prep`` is a hard dependency: it writes the four arrays that every other
stage loads. The remaining four are independent of each other and can be run in
any order.

``classify`` is by far the slowest — DTW k-NN and the two ROCKET variants each
cross-validate over 3,549 subjects — so budget hours for it. ``cluster`` is
next, dominated by the 3,549 x 3,549 pairwise DTW distance matrix.

Plots are shown interactively. When running headless, set a non-interactive
matplotlib backend first: ``MPLBACKEND=Agg python main.py --stage cluster``.
"""

import argparse
import warnings

from src.config import CLASS_NAMES, MODEL_CHANNEL_NAMES, N_CLUSTERS, PATHS
from src.utils import load_prepared_arrays, print_header, select_model_channels

# Heavier dependencies (aeon, sktime, tslearn, stumpy, PULSAR) are imported
# inside the stage that needs them, so running one stage does not require every
# library the project can use.

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# STEP 2 — Preparation
# ---------------------------------------------------------------------------

def run_preparation(show_plots=True):
    """Build the four prepared arrays from the raw pickle.

    Runs the full chain: load -> stratified split -> Hampel outlier filter ->
    interpolate -> Z-score normalise -> stack and save.

    Args:
        show_plots: Draw the three diagnostic figures. They are only useful
            interactively, so batch runs can turn them off.
    """
    from src.data_loader import (build_split, build_subject_metadata,
                                 describe_split, fit_feature_scalers,
                                 impute_missing, load_raw_dataset,
                                 normalize_subjects, remove_outliers_hampel,
                                 save_outlier_log, save_prepared_arrays,
                                 split_subjects)

    print_header("Data preparation")

    # --- Load and split -------------------------------------------------
    dataset = load_raw_dataset()
    print(f"Loaded {len(dataset)} subject time series.")

    subject_metadata = build_subject_metadata(dataset)
    train_metadata, test_metadata = split_subjects(subject_metadata)

    train_frames, y_train = build_split(dataset, train_metadata)
    test_frames, y_test = build_split(dataset, test_metadata)
    describe_split(y_train, y_test)
    print("-" * 50)
    print("Features remaining in train set:",
          train_frames[0].columns.tolist())

    # --- Missing values -------------------------------------------------
    total_missing = sum(frame.isna().sum().sum() for frame in dataset)
    print(f"\nTotal missing values across the entire dataset: {total_missing}")

    # --- Outliers, training set only ------------------------------------
    # The test subjects deliberately keep their raw values; see the module
    # docstring in src/data_loader.py.
    print("\nRunning the Hampel filter over the training subjects...")
    cleaned_frames, outlier_records = remove_outliers_hampel(train_frames)
    save_outlier_log(outlier_records)

    imputed_frames = impute_missing(cleaned_frames)

    # --- Normalisation --------------------------------------------------
    print("\nStacking training subjects to compute population baselines...")
    scalers = fit_feature_scalers(imputed_frames)
    print("Population baselines successfully calculated from the training "
          "data!\n")

    print("Normalizing training features across subjects...")
    train_normalized = normalize_subjects(imputed_frames, scalers)
    print("Normalizing test features across subjects (No Data Leakage)...")
    test_normalized = normalize_subjects(test_frames, scalers)
    print(f"Normalized Train Set: {len(train_normalized)} subjects")
    print(f"Normalized Test Set:  {len(test_normalized)} subjects\n")

    if show_plots:
        from src.plots import (plot_normalization_effect,
                               plot_outlier_treatment, plot_stl_decomposition)
        plot_outlier_treatment(train_frames, cleaned_frames, imputed_frames)
        plot_stl_decomposition(imputed_frames)
        plot_normalization_effect(imputed_frames, train_normalized)

    save_prepared_arrays(train_normalized, test_normalized, y_train, y_test)


# ---------------------------------------------------------------------------
# STEP 3 — Approximation, clustering, matrix profile
# ---------------------------------------------------------------------------

def run_clustering(show_plots=True):
    """Approximate the series, cluster them with DTW, and profile the centroids.

    Args:
        show_plots: Draw the k-search curves, the cluster projections and the
            centroid diagnostics.
    """
    from src.clustering import (analyze_centroid_matrix_profiles,
                                build_dtw_input, cluster_hierarchical_dtw,
                                cluster_kmeans_dtw, compute_cluster_centroids,
                                compute_dtw_distance_matrix, compute_paa,
                                plot_centroid_diagnostics,
                                plot_cluster_count_search,
                                plot_cluster_projections, project_pca,
                                project_tsne, search_cluster_count)

    print_header("Approximation, clustering and matrix profile")

    X_train, _, _, _ = load_prepared_arrays()
    print(f"Loaded a 3D array of shape {X_train.shape}.")

    _, paa_flat = compute_paa(X_train)

    # --- How many clusters? ---------------------------------------------
    inertia_values, silhouette_scores = search_cluster_count(paa_flat)
    if show_plots:
        plot_cluster_count_search(inertia_values, silhouette_scores)

    # --- Cluster with DTW -----------------------------------------------
    dtw_input = build_dtw_input(paa_flat)
    kmeans_labels = cluster_kmeans_dtw(dtw_input)
    distance_matrix = compute_dtw_distance_matrix(dtw_input)
    hierarchical_labels = cluster_hierarchical_dtw(distance_matrix)

    if show_plots:
        pca_points = project_pca(paa_flat)
        tsne_points = project_tsne(distance_matrix)
        plot_cluster_projections(pca_points, tsne_points, kmeans_labels,
                                 hierarchical_labels)

    # --- Matrix profile of each cluster's average shape -----------------
    centroids = compute_cluster_centroids(dtw_input, kmeans_labels)
    analyze_centroid_matrix_profiles(centroids)
    if show_plots:
        # Only cluster 0 is drawn; pass a different index to inspect another.
        plot_centroid_diagnostics(centroids, cluster_index=0)

    print(f"\nClustered {len(kmeans_labels)} subjects into {N_CLUSTERS} "
          "groups with both algorithms.")


# ---------------------------------------------------------------------------
# STEP 4 — Shapelets
# ---------------------------------------------------------------------------

def run_shapelets(show_plots=True):
    """Score the candidate shapelets and turn the winner into a rule.

    Args:
        show_plots: Draw the match-alignment figure for the top shapelet.
    """
    from src.shapelets import (compare_with_random_shapelets,
                               evaluate_shapelets,
                               extract_candidate_shapelets, extract_rule,
                               plot_shapelet_alignment, print_leaderboard,
                               subsequence_dist_euclidean)

    print_header("Shapelet analysis")

    X_train_raw, X_test_raw, y_train, y_test = load_prepared_arrays()
    X_train = select_model_channels(X_train_raw)
    X_test = select_model_channels(X_test_raw)
    print(f"Train set shape: {X_train.shape} (Subjects, Channels, Timesteps)")
    print(f"Test set shape:  {X_test.shape}")
    print(f"Training labels: {y_train.shape} "
          f"(Binary: 0={CLASS_NAMES[0]}, 1=Impaired)")

    shapelets = extract_candidate_shapelets(X_train, y_train)

    # A worked example of the distance definition before the full search.
    distance, match_index = subsequence_dist_euclidean(X_train[0, 0],
                                                       shapelets[0]["values"])
    print("\nSubject 0 - Subsequence alignment with Shapelet 0:")
    print(f"  Best match start index: {match_index}")
    print(f"  Minimum Euclidean Distance: {distance:.4f}\n")

    results = evaluate_shapelets(shapelets, X_train, y_train)
    ranked = print_leaderboard(results)

    print()
    compare_with_random_shapelets(X_train, y_train, ranked[0]["gain"])

    print()
    best_result, best_shapelet = extract_rule(ranked, shapelets)
    if show_plots:
        plot_shapelet_alignment(best_result, best_shapelet, X_train, y_train)


# ---------------------------------------------------------------------------
# STEP 5 — Classification
# ---------------------------------------------------------------------------

def run_classification(models=None, explain=True):
    """Train and evaluate the six classifiers, then compare them side by side.

    Args:
        models: Subset of model keys to run; defaults to all six. Skipping
            models also skips them in the final comparison table.
        explain: Run the PULSAR feature-importance and LORE explanations.
            Ignored when PULSAR itself is not run.
    """
    from src.metrics import print_model_comparison, print_naive_baseline
    from src.models import (add_jitter, extract_pulsar_features,
                            flatten_channels, run_knn_dtw, run_knn_euclidean,
                            run_minirocket, run_multirocket_hydra, run_pulsar,
                            run_rdst)

    print_header("Classification")

    X_train_raw, X_test_raw, y_train, y_test = load_prepared_arrays()
    X_train = select_model_channels(X_train_raw)
    X_test = select_model_channels(X_test_raw)
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape:  {X_test.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Modelled channels: {MODEL_CHANNEL_NAMES}\n")

    # 2-D views for the plain scikit-learn estimators, and jittered copies for
    # the aeon classifiers that reject zero-variance channels.
    X_train_flat = flatten_channels(X_train)
    X_test_flat = flatten_channels(X_test)
    print(f"Original shape:              {X_train.shape}")
    print(f"Flattened shape:             {X_train_flat.shape}\n")

    X_train_jittered, X_test_jittered = add_jitter(X_train, X_test)

    print_naive_baseline(y_train)

    selected = models or ["knn-euclidean", "knn-dtw", "rdst", "minirocket",
                          "mr-hydra", "pulsar"]
    results = []

    if "knn-euclidean" in selected:
        print_header("Model 1: KNN with Euclidean Distance")
        y_pred, y_prob = run_knn_euclidean(X_train_flat, y_train, X_test_flat,
                                           y_test)
        results.append(("KNN-Euclidean", y_pred, y_prob))

    if "knn-dtw" in selected:
        print_header("Model 2: KNN with DTW")
        y_pred, y_prob = run_knn_dtw(X_train, y_train, X_test, y_test)
        results.append(("KNN-DTW", y_pred, y_prob))

    if "rdst" in selected:
        print_header("Model 3: RDSTClassifier")
        y_pred, y_prob = run_rdst(X_train_jittered, y_train, X_test_jittered,
                                  y_test)
        results.append(("RDST", y_pred, y_prob))

    if "minirocket" in selected:
        print_header("Model 4: MiniRocket")
        y_pred, y_prob = run_minirocket(X_train_jittered, y_train,
                                        X_test_jittered, y_test)
        results.append(("MiniRocket", y_pred, y_prob))

    if "mr-hydra" in selected:
        print_header("Model 5: MultiRocket Hydra")
        y_pred, y_prob = run_multirocket_hydra(X_train_jittered, y_train,
                                               X_test_jittered, y_test)
        results.append(("MR-Hydra", y_pred, y_prob))

    if "pulsar" in selected:
        print_header("Model 6: Multivariate PULSAR (Early Fusion)")
        train_features, test_features, extractors, per_channel_train = \
            extract_pulsar_features(X_train, y_train, X_test)
        y_pred, y_prob, classifier = run_pulsar(train_features, test_features,
                                                y_train, y_test)
        results.append(("PULSAR", y_pred, y_prob))

        if explain:
            run_explainability(classifier, extractors, per_channel_train,
                               train_features, test_features)

    print_model_comparison(results, y_test)


def run_explainability(classifier, extractors, per_channel_train,
                       train_features, test_features):
    """Explain the fitted PULSAR classifier globally and locally.

    Args:
        classifier: The calibrated classifier returned by ``run_pulsar``.
        extractors: Fitted per-channel PULSAR extractors.
        per_channel_train: Per-channel training feature blocks.
        train_features: Concatenated training feature matrix.
        test_features: Concatenated test feature matrix.
    """
    from src.explainability import (build_feature_names, lore_explain,
                                    plot_feature_importances)

    print_header("Explainable AI for PULSAR")

    # The calibrated wrapper holds the actual forest; unwrap it to reach the
    # feature importances and to get plain (uncalibrated) predictions for LORE.
    forest = classifier.calibrated_classifiers_[0].estimator

    feature_names = build_feature_names(extractors, per_channel_train)
    assert len(feature_names) == train_features.shape[1], (
        f"Feature name count mismatch: {len(feature_names)} vs "
        f"{train_features.shape[1]}")

    plot_feature_importances(forest, feature_names)
    lore_explain(forest, train_features, test_features, feature_names)


# ---------------------------------------------------------------------------
# STEP 6 — Sequential pattern mining
# ---------------------------------------------------------------------------

def run_pattern_mining(show_plots=True):
    """Discretise the activity traces and mine contrasting behaviour patterns.

    Args:
        show_plots: Draw the contrast bar chart.
    """
    import os

    from src.config import SPM_CHANNEL
    from src.sequential_patterns import (discretize_sequences,
                                         plot_contrast_patterns,
                                         print_contrast_report,
                                         print_pattern_interpretation,
                                         print_symbol_distribution,
                                         report_global_patterns,
                                         run_contrast_mining)

    print_header("Sequential pattern mining")

    X_train, _, y_train, _ = load_prepared_arrays()
    # Sequential pattern mining uses the enmo channel alone: it is the direct
    # measure of how much a child is moving.
    enmo = X_train[:, :, SPM_CHANNEL]
    print(f"Loaded {enmo.shape[0]} subjects.\n")

    sequences, quantiles = discretize_sequences(enmo)
    print_symbol_distribution(sequences, y_train)

    print()
    report_global_patterns(sequences)

    print()
    unique_to_0, unique_to_1, support_0, support_1, _, _ = run_contrast_mining(
        sequences, y_train)
    top_class0, top_class1 = print_contrast_report(unique_to_0, unique_to_1,
                                                   support_0, support_1)

    if show_plots:
        os.makedirs(PATHS["figures_dir"], exist_ok=True)
        plot_contrast_patterns(
            top_class0, top_class1, support_0, support_1,
            os.path.join(PATHS["figures_dir"], "07_spm_contrast_patterns.png"))

    print("\nKey Insight:")
    print("- The Non-problematic group (SII=0) shows sequences with extended "
          "High activity (H -> H -> H -> M) and deeper Low-Medium cycling.")
    print("- The Problematic group (SII=1) is uniquely characterized by "
          "sustained unbroken chains of Sedentary/Low activity "
          "(L -> L -> L -> L -> L).")

    print_pattern_interpretation(top_class0, top_class1, support_0, support_1,
                                 quantiles)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

STAGES = ["prep", "cluster", "shapelets", "classify", "patterns"]

CLASSIFIER_KEYS = ["knn-euclidean", "knn-dtw", "rdst", "minirocket",
                   "mr-hydra", "pulsar"]


def main():
    """Parse arguments and run the requested stage or stages."""
    parser = argparse.ArgumentParser(
        description="Run the CMI actigraphy time-series pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", required=True, choices=STAGES + ["all"],
                        help="Which pipeline stage to run.")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=CLASSIFIER_KEYS,
                        help="classify stage only: limit to these models.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip every figure (useful for batch runs).")
    parser.add_argument("--no-explain", action="store_true",
                        help="classify stage only: skip the PULSAR feature "
                             "importances and LORE explanations.")
    args = parser.parse_args()

    show_plots = not args.no_plots
    stages = STAGES if args.stage == "all" else [args.stage]

    for stage in stages:
        print(f"\n{'#' * 70}\n# STAGE: {stage}\n{'#' * 70}")
        if stage == "prep":
            run_preparation(show_plots=show_plots)
        elif stage == "cluster":
            run_clustering(show_plots=show_plots)
        elif stage == "shapelets":
            run_shapelets(show_plots=show_plots)
        elif stage == "classify":
            run_classification(models=args.models,
                               explain=not args.no_explain)
        elif stage == "patterns":
            run_pattern_mining(show_plots=show_plots)


if __name__ == "__main__":
    main()
