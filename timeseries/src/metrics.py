"""
Evaluation metrics and plots for the classification stage.

**Why Macro F1 rather than accuracy.** Two thirds of the children are class 0,
so a model that always guesses "Healthy" scores 66.6% accuracy while finding
nobody at risk. Macro F1 averages the two classes' F1 scores with equal weight,
so ignoring the minority class is penalised properly. Every headline number in
this project is Macro F1; accuracy is reported alongside it for context only.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, roc_auc_score,
                             roc_curve)

from src.config import CLASS_NAMES, THRESHOLD_GRID


def print_naive_baseline(y):
    """Print the class balance and the accuracy of always guessing the majority.

    This is the number every model has to beat to be worth anything.

    Args:
        y: Label array.

    Returns:
        The majority-class percentage.
    """
    classes, counts = np.unique(y, return_counts=True)
    percentages = counts / len(y) * 100

    for class_value, count, percentage in zip(classes, counts, percentages):
        print(f"Class {class_value}: {count} subjects ({percentage:.2f}%)")

    benchmark = max(percentages)
    print(f"\nYour Naive Benchmark Accuracy is: {benchmark:.2f}%")
    return benchmark


def evaluate_model(model_name, y_true, y_pred, y_prob=None):
    """Report one model: full classification report, plots, then a summary.

    The report breaks precision and recall down per class, the confusion matrix
    shows exactly which children get misclassified in which direction, and the
    ROC curve summarises the ranking quality independently of any threshold.

    Args:
        model_name: Label for the printout and plot titles.
        y_true: True labels.
        y_pred: Predicted hard labels.
        y_prob: Predicted probability of class 1. When ``None`` the ROC panel
            is skipped and AUC is reported as N/A.
    """
    print(f"=== Evaluation: {model_name} ===")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    fig, (ax_matrix, ax_roc) = plt.subplots(1, 2, figsize=(14, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax_matrix, cmap="Blues",
        display_labels=CLASS_NAMES)
    ax_matrix.set_title(f"Confusion Matrix: {model_name}")

    if y_prob is not None:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax_roc.plot(false_positive_rate, true_positive_rate,
                    label=f"{model_name} (AUC = {auc:.2f})",
                    color="darkorange", lw=2)
    else:
        ax_roc.text(0.5, 0.5, "Probabilities not available", ha="center",
                    va="center")

    # The diagonal is what a coin flip would score.
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray",
                label="Random Guessing")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve: {model_name}")
    ax_roc.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    print(f"  [SUMMARY] {model_name}")
    print(f"    Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"    Macro F1:          "
          f"{f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"    Weighted F1:       "
          f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
    if y_prob is not None:
        try:
            print(f"    AUC-ROC:           "
                  f"{roc_auc_score(y_true, y_prob):.4f}")
        except Exception:
            print("    AUC-ROC:           N/A")
    print()


def find_optimal_threshold(y_true_cv, y_prob_cv):
    """Pick the probability cut-off that maximises Macro F1.

    The default 0.5 cut-off suits a balanced problem; here it under-predicts
    the minority class badly. Sweeping the threshold recovers recall on class 1
    at a small cost in precision.

    Crucially the sweep runs on **out-of-fold cross-validation predictions**,
    never on the test set — otherwise the chosen threshold would be tuned to
    the data used to report the final score, and the score would be optimistic.

    Args:
        y_true_cv: True training labels.
        y_prob_cv: Out-of-fold predicted probabilities of class 1.

    Returns:
        The best threshold found.
    """
    low, high, count = THRESHOLD_GRID
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.linspace(low, high, count):
        predictions = (y_prob_cv >= threshold).astype(int)
        score = f1_score(y_true_cv, predictions, average="macro")
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    print(f"Optimal CV Threshold Found: {best_threshold:.3f}  "
          f"(CV Macro F1: {best_f1:.3f})")
    return best_threshold


def print_model_comparison(model_results, y_test):
    """Print the final leaderboard of every model on the test set.

    Args:
        model_results: List of ``(name, y_pred, y_prob)`` tuples. ``y_prob``
            may be None for models without probability estimates.
        y_test: True test labels.
    """
    print("\n" + "=" * 70)
    print("=== FINAL MODEL COMPARISON — TEST SET ===")
    print("=" * 70)

    naive_accuracy = max(np.bincount(y_test.astype(int))) / len(y_test)
    print(f"  Naive baseline (majority class accuracy): {naive_accuracy:.4f}")
    print()
    print(f"  {'Model':<18} | {'Accuracy':>10} | {'Macro F1':>10} | "
          f"{'AUC-ROC':>10}")
    print(f"  {'-' * 18}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    for name, y_pred, y_prob in model_results:
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        try:
            auc = (roc_auc_score(y_test, y_prob) if y_prob is not None
                   else float("nan"))
        except Exception:
            auc = float("nan")
        print(f"  {name:<18} | {accuracy:>10.4f} | {macro_f1:>10.4f} | "
              f"{auc:>10.4f}")

    print("=" * 70)
