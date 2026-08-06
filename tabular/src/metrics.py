"""
STEP 5 — Evaluation metrics, threshold tuning and plots.

The headline metric differs between the two tracks:

* Regression predicts `sii` as a continuous number, so its output has to be cut
  into the four ordinal classes before it can be scored. The cut points are
  themselves tuned (see the threshold functions below) and the result is scored
  with **Quadratic Weighted Kappa** (QWK), which penalises being three classes
  wrong far more than being one class wrong.
* Classification predicts the four classes directly and is scored with
  **Macro F1**, which weights all four classes equally despite ~69% of children
  sitting in class 0.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (auc, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from src.config import SEEDS, SII_CLASSES

# Starting cut points for the ordinal thresholds: the midpoints between the
# four class labels 0, 1, 2, 3.
INITIAL_THRESHOLDS = [0.5, 1.5, 2.5]


def quadratic_weighted_kappa(y_true, y_pred):
    """Agreement between two sets of ordinal labels, penalising by squared distance.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Float in [-1, 1]; 0 means chance agreement, 1 means perfect.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# ---------------------------------------------------------------------------
# Turning continuous regression output into ordinal classes
# ---------------------------------------------------------------------------

def apply_thresholds(predictions, thresholds):
    """Cut continuous predictions into the classes {0, 1, 2, 3}.

    A prediction lands in class *k* where *k* is the number of thresholds it
    exceeds, so ``thresholds`` must be sorted ascending.

    Args:
        predictions: 1D array of continuous predictions.
        thresholds: Three ascending cut points ``[t1, t2, t3]``.

    Returns:
        1D integer array of class labels.
    """
    predictions = np.asarray(predictions)
    labels = np.zeros_like(predictions, dtype=int)
    for threshold in thresholds:
        labels += (predictions > threshold).astype(int)
    return np.clip(labels, 0, 3)


def optimize_thresholds_grid(y_true, predictions, init_thresholds=None,
                             span=0.5, step=0.05, n_iter=3):
    """Find QWK-maximising cut points by repeated local grid search.

    Each pass sweeps one threshold at a time over ``[current +/- span]`` in
    ``step`` increments, keeping any candidate that improves QWK and rejecting
    any that would break the ascending order. Used inside cross-validation,
    where it is fitted on the training fold only.

    Args:
        y_true: True ordinal labels.
        predictions: Continuous predictions.
        init_thresholds: Starting cut points; defaults to ``[0.5, 1.5, 2.5]``.
        span: Half-width of the window searched around each threshold.
        step: Spacing between candidate values.
        n_iter: Number of refinement passes over all three thresholds.

    Returns:
        List of three optimised thresholds.
    """
    predictions = np.asarray(predictions)
    thresholds = list(init_thresholds or INITIAL_THRESHOLDS)
    lowest, highest = predictions.min(), predictions.max()

    best_thresholds = thresholds.copy()
    best_score = quadratic_weighted_kappa(
        y_true, apply_thresholds(predictions, best_thresholds)
    )

    for _ in range(n_iter):
        for index in range(len(thresholds)):
            window_low = max(lowest, thresholds[index] - span)
            window_high = min(highest, thresholds[index] + span)
            for value in np.arange(window_low, window_high + 1e-9, step):
                candidate = thresholds.copy()
                candidate[index] = value
                if not candidate[0] < candidate[1] < candidate[2]:
                    continue
                score = quadratic_weighted_kappa(
                    y_true, apply_thresholds(predictions, candidate)
                )
                if score > best_score:
                    best_score = score
                    best_thresholds = candidate.copy()
            thresholds[index] = best_thresholds[index]

    return best_thresholds


def optimize_thresholds_nelder_mead(y_true, predictions):
    """Find QWK-maximising cut points with the Nelder-Mead simplex method.

    Faster and less granular than the grid search; used at final evaluation,
    where the thresholds are fitted on the training set and then applied
    unchanged to the test set.

    Args:
        y_true: True ordinal labels.
        predictions: Continuous predictions.

    Returns:
        numpy array of three optimised thresholds.
    """
    predictions = np.asarray(predictions)

    def negative_kappa(cut_points):
        # Reject any simplex point that puts the thresholds out of order.
        if cut_points[0] >= cut_points[1] or cut_points[1] >= cut_points[2]:
            return 1.0
        labels = np.zeros_like(predictions)
        labels[predictions > cut_points[0]] = 1
        labels[predictions > cut_points[1]] = 2
        labels[predictions > cut_points[2]] = 3
        return -quadratic_weighted_kappa(y_true, labels)

    return minimize(negative_kappa, INITIAL_THRESHOLDS,
                    method="Nelder-Mead").x


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, title, cmap="Blues"):
    """Draw a confusion matrix heatmap with raw counts."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap=cmap)
    plt.title(title)
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.show()


def evaluate_classifier(name, model, X_test, y_test, probabilities,
                        class_names=None, y_test_binary=None):
    """Print a full classification report and plot its confusion matrix and ROC.

    ROC curves are drawn one-vs-rest: one curve per class, treating that class
    as positive and the other three as negative.

    Args:
        name: Display name for the model.
        model: Fitted estimator exposing ``predict``.
        X_test: Test features.
        y_test: True test labels (pandas Series).
        probabilities: Predicted probabilities, shape (n_samples, n_classes).
        class_names: Optional display names for the four classes.
        y_test_binary: Optional pre-computed one-hot labels for the ROC curves.
    """
    classes = sorted(y_test.unique())
    if y_test_binary is None:
        y_test_binary = label_binarize(y_test, classes=classes)

    print(f"\n{'=' * 60}\nEvaluation for {name}\n{'=' * 60}")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=class_names))

    try:
        macro_auc = roc_auc_score(y_test_binary, probabilities,
                                  multi_class="ovr", average="macro")
        print(f"Macro AUC: {macro_auc:.4f}")
    except Exception as error:
        print(f"Could not calculate AUC: {error}")

    figure, (confusion_axis, roc_axis) = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d",
                cmap="Blues", ax=confusion_axis, cbar=False)
    confusion_axis.set_title(f"{name} - Confusion Matrix")
    confusion_axis.set_xlabel("Predicted Label")
    confusion_axis.set_ylabel("True Label")

    for column, class_label in enumerate(classes):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test_binary[:, column], probabilities[:, column]
        )
        roc_axis.plot(false_positive_rate, true_positive_rate, lw=2,
                      label=f"Class {class_label} "
                            f"(AUC = {auc(false_positive_rate, true_positive_rate):.2f})")

    roc_axis.plot([0, 1], [0, 1], "k--", lw=2)  # chance line
    roc_axis.set_xlim([0.0, 1.0])
    roc_axis.set_ylim([0.0, 1.05])
    roc_axis.set_xlabel("False Positive Rate")
    roc_axis.set_ylabel("True Positive Rate")
    roc_axis.set_title(f"{name} - ROC Curve (OvR)")
    roc_axis.legend(loc="lower right")
    plt.show()


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_regressors(X_train, y_train, X_test, y_test, models):
    """Fit each regressor, tune its cut points on train, and score it on test.

    Thresholds are always tuned on *training* predictions and then applied
    unchanged to the test predictions — tuning them on the test set would leak
    the answer into the metric.

    Args:
        X_train: Preprocessed training features.
        y_train: Training labels.
        X_test: Preprocessed test features.
        y_test: Test labels.
        models: Dict of ``{display_name: estimator}``.

    Returns:
        Dict keyed by model name, each holding ``Test_QWK``, ``Thresholds``
        and ``Test_Class_Preds``.
    """
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        thresholds = optimize_thresholds_nelder_mead(
            y_train, model.predict(X_train)
        )
        class_predictions = apply_thresholds(model.predict(X_test), thresholds)
        test_qwk = quadratic_weighted_kappa(y_test, class_predictions)

        results[name] = {
            "Test_QWK": test_qwk,
            "Thresholds": thresholds,
            "Test_Class_Preds": class_predictions,
        }
        print(f" -> Test QWK: {test_qwk:.4f}")
    return results


def compare_regressor_results(results):
    """Print a leaderboard of test QWK, best first.

    Args:
        results: Output of :func:`train_and_evaluate_regressors`.

    Returns:
        pd.DataFrame sorted by test QWK descending.
    """
    leaderboard = pd.DataFrame([
        {"Model": name, "Test QWK": scores["Test_QWK"]}
        for name, scores in results.items()
    ]).sort_values("Test QWK", ascending=False).reset_index(drop=True)

    print("=== Base Models Test Set Performance ===")
    print(leaderboard.to_string(index=False))
    return leaderboard


def build_regression_stack(X_train, y_train, X_test, y_test, base_estimators):
    """Fit a stacking regressor and score it with freshly tuned thresholds.

    A ``RidgeCV`` meta-learner is fitted on the base models' out-of-fold
    predictions (5-fold internally), which is what lets the ensemble learn
    where each base model is trustworthy. Because the ensemble's output
    distribution differs from any single model's, its cut points are tuned
    from scratch rather than reused.

    Args:
        X_train: Preprocessed training features.
        y_train: Training labels.
        X_test: Preprocessed test features.
        y_test: Test labels.
        base_estimators: List of ``(name, estimator)`` tuples.

    Returns:
        Tuple ``(fitted_stacking_model, results_dict)``.
    """
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    print("Fitting Stacking Model...")
    stacking_model.fit(X_train, y_train)

    thresholds = optimize_thresholds_nelder_mead(
        y_train, stacking_model.predict(X_train)
    )
    class_predictions = apply_thresholds(stacking_model.predict(X_test),
                                         thresholds)
    test_qwk = quadratic_weighted_kappa(y_test, class_predictions)

    print("\n=== Stacking Ensemble Performance ===")
    print(f"Stacking Test QWK: {test_qwk:.4f}")

    plot_confusion_matrix(y_test, class_predictions,
                          "Test Set Confusion Matrix - Stacking Ensemble",
                          cmap="Greens")

    results = {"Stacking": {
        "Test_QWK": test_qwk,
        "Thresholds": thresholds,
        "Test_Class_Preds": class_predictions,
    }}
    return stacking_model, results


def interpret_meta_weights(stacking_model):
    """Show how much the meta-learner leans on each base model.

    A large positive coefficient means the ensemble trusts that base model; a
    negative one means it is used as a corrective counterweight.

    Args:
        stacking_model: A fitted ``StackingRegressor``.
    """
    meta_estimator = stacking_model.final_estimator_
    if not hasattr(meta_estimator, "coef_"):
        print("The final estimator does not expose linear coefficients.")
        return

    weights = meta_estimator.coef_.ravel()
    model_names = list(stacking_model.named_estimators_.keys())
    if len(model_names) != len(weights):
        print(f"Dimension mismatch: {len(model_names)} models vs "
              f"{len(weights)} coefficients.")
        return

    weight_table = pd.DataFrame({
        "Base Model": model_names,
        "Meta Weight": weights,
    }).sort_values("Meta Weight", ascending=False)

    print("=== Meta-Model Coefficients ===")
    print(weight_table.to_string(index=False))

    plt.figure(figsize=(8, 4))
    sns.barplot(data=weight_table, x="Meta Weight", y="Base Model",
                hue="Base Model", palette="viridis", legend=False)
    plt.title("Stacking Ensemble: Meta-Model Coefficients")
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Classification stacking
# ---------------------------------------------------------------------------

def build_classification_stack(X_train, y_train, X_test, y_test,
                               base_estimators, class_names=None):
    """Fit a stacking classifier and rescale its probabilities for Macro F1.

    Argmax over raw probabilities optimises accuracy, which on this dataset
    just means predicting the majority class. So after fitting, we search for
    one multiplier per class (Nelder-Mead, clipped to [0.1, 10]) that maximises
    Macro F1 on the *training* predictions, then apply those same multipliers
    to the test probabilities. In practice the search boosts the rare classes
    and shrinks class 0.

    Args:
        X_train: Preprocessed training features.
        y_train: Training labels.
        X_test: Preprocessed test features.
        y_test: Test labels.
        base_estimators: List of ``(name, estimator)`` tuples.
        class_names: Optional display names for the classification report.

    Returns:
        Tuple ``(fitted_stacker, test_predictions, multipliers, test_probabilities)``.
    """
    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                           random_state=SEEDS["random_state"]),
        cv=5,
        stack_method="predict_proba",
        n_jobs=1,
        passthrough=False,
    )

    print("Fitting Stacking Ensemble (this may take a few minutes)...")
    stacker.fit(X_train, y_train)

    print("Optimizing probability thresholds directly for Macro F1...")
    train_probabilities = stacker.predict_proba(X_train)

    def negative_macro_f1(multipliers):
        multipliers = np.clip(multipliers, 0.1, 10.0)
        predictions = np.argmax(train_probabilities * multipliers, axis=1)
        return -f1_score(y_train, predictions, average="macro")

    search = minimize(negative_macro_f1, [1.0] * len(SII_CLASSES),
                      method="Nelder-Mead", options={"maxiter": 500})
    multipliers = np.clip(search.x, 0.1, 10.0)
    print(f"Optimal Class Multipliers: {multipliers}")

    test_probabilities = stacker.predict_proba(X_test)
    test_predictions = np.argmax(test_probabilities * multipliers, axis=1)

    print("\n--- Stacking Ensemble Performance (Macro F1 Optimized) ---")
    print(classification_report(y_test, test_predictions,
                                target_names=class_names))
    print(f"Log Loss: {log_loss(y_test, test_probabilities):.4f}")
    print(f"QWK: {quadratic_weighted_kappa(y_test, test_predictions):.4f}")

    plot_confusion_matrix(y_test, test_predictions,
                          "Confusion Matrix: Threshold Optimized Stacking")

    return stacker, test_predictions, multipliers, test_probabilities
