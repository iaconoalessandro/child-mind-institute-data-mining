"""
STEP 5 — Classification models.

Six classifiers, each representing a different idea of what makes two activity
traces similar:

============================  =======================================
Model                         What it looks at
============================  =======================================
k-NN (Euclidean)              the raw 400-number vector, point by point
k-NN (DTW)                    the same shape, allowing it to drift in time
RDST                          which short shapelets appear in the series
MiniRocket                    responses to 10,000 fixed convolution kernels
MultiRocket-Hydra             the same idea, with several kernel families
PULSAR                        pooled statistics of random sub-intervals
============================  =======================================

Only k-NN (Euclidean) is hyperparameter-tuned. The other five run at their
published defaults, because tuning a DTW or shapelet model over 3,549 subjects
costs many hours per study for a fraction of a point of F1.

Three library-level fixes are applied and documented at their call sites: a
Sakoe-Chiba band so DTW terminates, Gaussian jitter so the aeon classifiers
tolerate flat channels, and a raised iteration cap so MiniRocket's internal
logistic regression converges.
"""

import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (CV_FOLDS, DTW_WINDOW_FRACTION, JITTER_SIGMA,
                        KNN_N_NEIGHBORS_RANGE, KNN_OPTUNA_TRIALS,
                        MINIROCKET_MAX_ITER, MINIROCKET_N_KERNELS,
                        PULSAR_ET_PARAMS, PULSAR_PARAMS, RDST_MAX_SHAPELETS,
                        SEEDS)
from src.metrics import evaluate_model, find_optimal_threshold
from src.utils import add_pulsar_to_path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def flatten_channels(X):
    """Collapse the channel and time axes into one long feature vector.

    Needed by the plain scikit-learn estimators, which expect a 2-D table:
    ``(n_subjects, 2, 200)`` becomes ``(n_subjects, 400)``.

    Args:
        X: Array of shape ``(n_subjects, n_channels, n_timesteps)``.

    Returns:
        2-D array of shape ``(n_subjects, n_channels * n_timesteps)``.
    """
    n_subjects, n_channels, n_timesteps = X.shape
    return X.reshape(n_subjects, n_channels * n_timesteps)


def add_jitter(X_train, X_test):
    """Add imperceptible Gaussian noise to break perfectly flat channels.

    ``RDSTClassifier`` aborts with a fatal ``ValueError (std <= 1e-07)`` if any
    subject has a completely constant channel. A jitter of sigma = 1e-6 sits far
    below the resolution of the normalised signal but satisfies that internal
    variance check.

    Both arrays are drawn from one generator, in this order, so the noise is
    reproducible across runs.

    Args:
        X_train: Training array.
        X_test: Test array.

    Returns:
        Tuple ``(X_train_jittered, X_test_jittered)``.
    """
    rng = np.random.default_rng(seed=SEEDS["jitter"])
    return (X_train + rng.normal(0, JITTER_SIGMA, X_train.shape),
            X_test + rng.normal(0, JITTER_SIGMA, X_test.shape))


def tune_threshold_with_cv(model, X_train, y_train, model_label):
    """Get out-of-fold probabilities and pick the best decision threshold.

    ``cross_val_predict`` gives every training subject a prediction from a fold
    that did not see it, which is what makes the resulting threshold honest.

    Failures are caught rather than raised: some of these classifiers are heavy
    and can run out of memory mid-fold, and losing the threshold search should
    not lose the rest of the run — the model then falls back to 0.5.

    Args:
        model: Unfitted estimator exposing ``predict_proba``.
        X_train: Training features in whatever shape the model expects.
        y_train: Training labels.
        model_label: Name used in the failure message.

    Returns:
        The chosen threshold, or 0.5 if cross-validation failed.
    """
    best_threshold = 0.5
    try:
        print(f"Computing {CV_FOLDS}-fold cross-validation out-of-fold "
              "predictions...")
        cv_probabilities = cross_val_predict(model, X_train, y_train,
                                             cv=CV_FOLDS,
                                             method="predict_proba")
        class1_probabilities = (cv_probabilities[:, 1]
                                if cv_probabilities.ndim == 2
                                else cv_probabilities[1, :])
        best_threshold = find_optimal_threshold(y_train, class1_probabilities)

        print("\nCross-Validation Classification Report (Optimal Threshold):")
        print(classification_report(
            y_train, (class1_probabilities >= best_threshold).astype(int)))
    except Exception as error:
        print(f"Cross-validation failed for {model_label}: {error}")

    return best_threshold


def fit_and_predict(model, X_train, y_train, X_test, threshold):
    """Fit a model on the full training set and apply the tuned threshold.

    Args:
        model: Unfitted estimator.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        threshold: Probability cut-off from :func:`tune_threshold_with_cv`.

    Returns:
        Tuple ``(y_pred, y_prob)`` for the test set.
    """
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return (y_prob >= threshold).astype(int), y_prob


# ---------------------------------------------------------------------------
# Model 1 — k-NN with Euclidean distance
# ---------------------------------------------------------------------------

def run_knn_euclidean(X_train_flat, y_train, X_test_flat, y_test):
    """Tune, cross-validate and evaluate a Euclidean k-nearest-neighbours model.

    The simplest possible baseline: compare two children by lining their
    activity traces up point for point. It has no notion that the same routine
    an hour later is still the same routine — which is exactly the weakness the
    DTW version below is meant to fix.

    ``weights='distance'`` lets closer neighbours count for more. The scaler
    lives inside the pipeline so it is refitted within every CV fold rather
    than once over all the data.

    Args:
        X_train_flat: Flattened training features.
        y_train: Training labels.
        X_test_flat: Flattened test features.
        y_test: Test labels.

    Returns:
        Tuple ``(y_pred, y_prob)`` for the test set.
    """
    def build(n_neighbors):
        """Build the scaler + k-NN pipeline for a given neighbour count."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors,
                                         metric="euclidean",
                                         weights="distance")),
        ])

    def objective(trial):
        """Optuna objective: mean cross-validated Macro F1 for one k."""
        n_neighbors = trial.suggest_int("n_neighbors", *KNN_N_NEIGHBORS_RANGE)
        return cross_val_score(build(n_neighbors), X_train_flat, y_train,
                               cv=CV_FOLDS, scoring="f1_macro").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=KNN_OPTUNA_TRIALS)
    best_params = study.best_params
    print(f"Best hyperparameters for KNN (Euclidean): {best_params}")

    model = build(best_params["n_neighbors"])

    cv_predictions = cross_val_predict(model, X_train_flat, y_train,
                                       cv=CV_FOLDS)
    print("\nCross-Validation Classification Report:")
    print(classification_report(y_train, cv_predictions))

    # This model uses the default 0.5 cut-off rather than a tuned threshold.
    model.fit(X_train_flat, y_train)
    y_pred = model.predict(X_test_flat)
    y_prob = model.predict_proba(X_test_flat)[:, 1]

    evaluate_model("KNN (Euclidean)", y_test, y_pred, y_prob)
    print("\n=== KNN (Euclidean) — hyperparameter tuning result ===")
    print(f"  Best n_neighbors: {best_params['n_neighbors']}")
    print(f"  Best CV Macro F1: {study.best_value:.4f}")
    print(f"  (Across {len(study.trials)} Optuna trials, {CV_FOLDS}-fold CV)")

    return y_pred, y_prob


# ---------------------------------------------------------------------------
# Model 2 — k-NN with Dynamic Time Warping
# ---------------------------------------------------------------------------

def run_knn_dtw(X_train, y_train, X_test, y_test):
    """Cross-validate and evaluate a DTW-based k-nearest-neighbours model.

    DTW stretches and compresses the time axis to find the best alignment
    between two series, so two children with the same routine shifted by an
    hour still count as similar.

    The Sakoe-Chiba band is not optional: unbounded DTW is O(n^2) per pair and
    hangs indefinitely on thousands of multivariate samples. Restricting the
    warping corridor to 10% of the series length makes it tractable.

    Args:
        X_train: Training array ``(n_subjects, 2, n_timesteps)``.
        y_train: Training labels.
        X_test: Test array.
        y_test: Test labels.

    Returns:
        Tuple ``(y_pred, y_prob)`` for the test set.
    """
    from sktime.classification.distance_based import \
        KNeighborsTimeSeriesClassifier

    model = KNeighborsTimeSeriesClassifier(
        distance="dtw",
        n_jobs=-1,
        distance_params={"window": DTW_WINDOW_FRACTION},
    )

    print("Skipping Optuna tuning for KNN with DTW due to computational cost. "
          f"Using Sakoe-Chiba-bounded DTW (window={DTW_WINDOW_FRACTION}).")

    threshold = tune_threshold_with_cv(model, X_train, y_train, "KNN DTW")
    y_pred, y_prob = fit_and_predict(model, X_train, y_train, X_test,
                                     threshold)
    evaluate_model("KNN (DTW)", y_test, y_pred, y_prob)

    print("=== KNN (DTW) — key parameters and final test results ===")
    print(f"  DTW window (Sakoe-Chiba): "
          f"{DTW_WINDOW_FRACTION:.0%} of series length = "
          f"{int(DTW_WINDOW_FRACTION * X_train.shape[2])} steps")
    print(f"  Optimal CV threshold:     {threshold:.3f}")

    return y_pred, y_prob


# ---------------------------------------------------------------------------
# Model 3 — RDST (Random Dilated Shapelet Transform)
# ---------------------------------------------------------------------------

def run_rdst(X_train_jittered, y_train, X_test_jittered, y_test):
    """Cross-validate and evaluate the Random Dilated Shapelet Transform.

    RDST samples many short shapelets at random, with *dilation* — gaps between
    the sampled points — so one short shapelet can describe a pattern spread
    over a long stretch of time. Each series is then described by how well it
    matches each shapelet.

    This is the automated counterpart to the hand-picked shapelet search in
    ``src/shapelets.py``.

    Args:
        X_train_jittered: Jittered training array.
        y_train: Training labels.
        X_test_jittered: Jittered test array.
        y_test: Test labels.

    Returns:
        Tuple ``(y_pred, y_prob)`` for the test set.
    """
    from aeon.classification.shapelet_based import RDSTClassifier

    model = RDSTClassifier(max_shapelets=RDST_MAX_SHAPELETS, n_jobs=-1)
    print("Skipping Optuna tuning for RDSTClassifier due to computational "
          "cost. Using default parameters + jitter fix.")

    threshold = tune_threshold_with_cv(model, X_train_jittered, y_train,
                                       "RDSTClassifier")
    y_pred, y_prob = fit_and_predict(model, X_train_jittered, y_train,
                                     X_test_jittered, threshold)
    evaluate_model("RDSTClassifier", y_test, y_pred, y_prob)

    return y_pred, y_prob


# ---------------------------------------------------------------------------
# Model 4 — MiniRocket
# ---------------------------------------------------------------------------

def run_minirocket(X_train_jittered, y_train, X_test_jittered, y_test):
    """Cross-validate and evaluate MiniRocket.

    MiniRocket convolves each series with 10,000 small fixed kernels and
    summarises the responses. Because the kernels are fixed rather than learned,
    the transform is extremely fast, and a plain linear classifier on top is
    usually enough.

    That linear classifier needs help: the ROCKET feature space is
    high-dimensional, so the default 1,000 iterations leave it unconverged.
    Scaling the features (without centring, to keep the sparse structure) and
    raising the cap to 5,000 fixes it.

    Args:
        X_train_jittered: Jittered training array.
        y_train: Training labels.
        X_test_jittered: Jittered test array.
        y_test: Test labels.

    Returns:
        Tuple ``(y_pred, y_prob)`` for the test set.
    """
    from aeon.classification.convolution_based import MiniRocketClassifier

    model = MiniRocketClassifier(
        n_kernels=MINIROCKET_N_KERNELS,
        n_jobs=-1,
        estimator=make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(penalty="l2", class_weight="balanced",
                               max_iter=MINIROCKET_MAX_ITER),
        ),
    )

    print("Skipping Optuna tuning for MiniRocket (default parameters are "
          "highly optimised + convergence fix applied).")

    # This model is scored on its hard labels, so no threshold search is run.
    try:
        print(f"Computing {CV_FOLDS}-fold cross-validation out-of-fold "
              "predictions...")
        cv_predictions = cross_val_predict(model, X_train_jittered, y_train,
                                           cv=CV_FOLDS)
        print("\nCross-Validation Classification Report:")
        print(classification_report(y_train, cv_predictions))
    except Exception as error:
        print(f"Cross-validation failed for MiniRocket: {error}")

    model.fit(X_train_jittered, y_train)
    y_pred = model.predict(X_test_jittered)
    try:
        y_prob = model.predict_proba(X_test_jittered)[:, 1]
    except AttributeError:
        y_prob = None

    evaluate_model("MiniRocket", y_test, y_pred, y_prob)
    return y_pred, y_prob


# ---------------------------------------------------------------------------
# Model 5 — MultiRocket-Hydra
# ---------------------------------------------------------------------------

def run_multirocket_hydra(X_train_jittered, y_train, X_test_jittered, y_test):
    """Cross-validate and evaluate MultiRocket-Hydra.

    Hydra combines two ideas: MultiRocket's several kernel families (which
    capture more than just "did this kernel fire") and a dictionary-style count
    of which kernel wins in each neighbourhood. It is one of the strongest
    general-purpose time-series classifiers available.

    Args:
        X_train_jittered: Jittered training array.
        y_train: Training labels.
        X_test_jittered: Jittered test array.
        y_test: Test labels.

    Returns:
        Tuple ``(y_pred, y_prob)`` for the test set.
    """
    from aeon.classification.convolution_based import \
        MultiRocketHydraClassifier

    model = MultiRocketHydraClassifier(n_jobs=-1)
    print("Skipping Optuna tuning for MultiRocket Hydra due to computational "
          "cost.")

    threshold = tune_threshold_with_cv(model, X_train_jittered, y_train,
                                       "MultiRocket Hydra")

    model.fit(X_train_jittered, y_train)
    try:
        y_prob = model.predict_proba(X_test_jittered)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    except AttributeError:
        # Fall back to hard labels if this build exposes no probabilities.
        y_prob = None
        y_pred = model.predict(X_test_jittered)

    evaluate_model("MultiRocket Hydra", y_test, y_pred, y_prob)
    return y_pred, y_prob


# ---------------------------------------------------------------------------
# Model 6 — Multivariate PULSAR (early fusion)
# ---------------------------------------------------------------------------

def build_unweighted_pulsar():
    """Return a PULSAR subclass that extracts features but does not classify.

    PULSAR normally trains its own internal ensemble. Here it is wanted purely
    as a feature extractor, so that step is overridden away and a single
    classifier is trained on both channels at once instead.

    Returns:
        The ``UnweightedPULSAR`` class (imported lazily, since PULSAR lives
        outside the package).
    """
    add_pulsar_to_path()
    from pulsar import PULSAR

    class UnweightedPULSAR(PULSAR):
        """PULSAR with its internal classifier initialisation disabled."""

        def initialise_classifiers(self, X, y, cv):
            """Do nothing — classification is handled externally."""

    return UnweightedPULSAR


def extract_pulsar_features(X_train, y_train, X_test):
    """Fit one PULSAR extractor per channel and concatenate their features.

    This is *early fusion*: rather than classifying each channel and combining
    votes afterwards, the per-channel feature blocks are glued into one wide
    matrix so the classifier can learn interactions **between** enmo and
    anglez directly.

    Each extractor is fitted on training data only, then applied unchanged to
    the test data.

    Args:
        X_train: Training array ``(n_subjects, 2, n_timesteps)``.
        y_train: Training labels.
        X_test: Test array.

    Returns:
        Tuple ``(train_features, test_features, extractors, per_channel_train)``
        where the feature matrices are 2-D and ``extractors`` is the list of
        fitted per-channel extractors.
    """
    unweighted_pulsar = build_unweighted_pulsar()
    n_channels = X_train.shape[1]

    print("Initializing PULSAR Feature Extractors...")
    print("Extracting features from all channels for Multivariate PULSAR...")

    extractors = []
    per_channel_train = []
    for channel in range(n_channels):
        print(f"  -> Fitting extractor and transforming Channel "
              f"{channel}/{n_channels - 1}...")
        extractor = unweighted_pulsar(**PULSAR_PARAMS)
        extractor.fit(X_train[:, channel, :], y_train)
        per_channel_train.append(extractor.transform(X_train[:, channel, :]))
        extractors.append(extractor)

    train_features = np.hstack(per_channel_train)
    print(f"\nMultivariate Training Matrix Shape: {train_features.shape}")

    print("Extracting test features and evaluating...")
    test_features = np.hstack([
        extractors[channel].transform(X_test[:, channel, :])
        for channel in range(n_channels)
    ])

    return train_features, test_features, extractors, per_channel_train


def run_pulsar(train_features, test_features, y_train, y_test):
    """Train and evaluate the calibrated ExtraTrees on PULSAR's features.

    ExtraTrees suits a very wide feature matrix: it picks split points at
    random, which is fast and resists overfitting when there are far more
    features than subjects. ``class_weight='balanced'`` compensates for the 2:1
    class imbalance.

    The calibration wrapper converts raw tree votes into usable probabilities,
    which the threshold search needs.

    Args:
        train_features: PULSAR feature matrix for training subjects.
        test_features: PULSAR feature matrix for test subjects.
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        Tuple ``(y_pred, y_prob, fitted_classifier)``.
    """
    cross_val_calibrator = CalibratedClassifierCV(
        ExtraTreesClassifier(**PULSAR_ET_PARAMS), method="sigmoid",
        cv=CV_FOLDS)

    print()
    threshold = tune_threshold_with_cv(cross_val_calibrator, train_features,
                                       y_train, "Multivariate PULSAR")

    # The final model is built fresh: fit the forest, then calibrate it.
    print("Fitting final multivariate classifier...")
    forest = ExtraTreesClassifier(**PULSAR_ET_PARAMS)
    forest.fit(train_features, y_train)
    classifier = CalibratedClassifierCV(forest, method="sigmoid", cv="prefit")
    classifier.fit(train_features, y_train)

    y_prob = classifier.predict_proba(test_features)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    evaluate_model("Multivariate PULSAR (Early Fusion)", y_test, y_pred,
                   y_prob)
    return y_pred, y_prob, classifier
