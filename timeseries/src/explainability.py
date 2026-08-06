"""
Explaining the PULSAR classifier.

Two complementary views of the same fitted model:

* **Global** — :func:`plot_feature_importances` asks which of the ~13,000
  PULSAR features the forest relies on across all subjects.
* **Local** — :func:`lore_explain` asks why *this particular child* was
  classified the way they were, and what would have had to differ for the
  answer to flip.

LORE (Local Rule-based Explanations) works by generating a cloud of synthetic
neighbours around one instance, asking the black-box model to label them, and
fitting a small decision tree to that cloud. The tree is only trusted locally,
and its *fidelity* — how often it agrees with the black box on the cloud — is
reported so a poor approximation is visible rather than hidden.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from src.config import (CLASS_NAMES, LORE_EPSILON, LORE_EXPLAIN_INDICES,
                        LORE_N_NEIGHBOURS, LORE_NOISE_SCALE, LORE_TREE_MAX_DEPTH,
                        SEEDS)


def build_feature_names(extractors, per_channel_train):
    """Reconstruct readable names for PULSAR's concatenated feature columns.

    PULSAR records what each feature was built from; this turns that record
    into names like ``ch0_original_iqr_stdev_pooling`` — channel, series
    representation, local statistic, pooling operator. Without them the
    explanations below would only be able to cite column numbers.

    Args:
        extractors: Fitted per-channel PULSAR extractors.
        per_channel_train: Per-channel training feature blocks, used only to
            get column counts for the fallback naming.

    Returns:
        List of feature names, in the same order as the concatenated matrix.
    """
    feature_names = []

    for channel, extractor in enumerate(extractors):
        if hasattr(extractor, "total_list_of_indices_combined"):
            for feature in extractor.total_list_of_indices_combined:
                representation = feature[1]
                statistic = feature[9] if len(feature) > 9 and feature[9] else "raw"
                pooling = feature[8] if len(feature) > 8 and feature[8] else "raw"
                feature_names.append(
                    f"ch{channel}_{representation}_{statistic}_{pooling}")
        else:
            # Older PULSAR builds do not expose the index list; fall back to
            # positional names so the column count still lines up.
            n_features = per_channel_train[channel].shape[1]
            feature_names.extend(
                [f"ch{channel}_feature_{i}" for i in range(n_features)])

    return feature_names


def plot_feature_importances(forest, feature_names, top_n=15):
    """Plot the features the forest splits on most often.

    Args:
        forest: A fitted tree ensemble exposing ``feature_importances_``.
        feature_names: Names aligned with the model's columns.
        top_n: How many features to show.
    """
    importances = pd.Series(forest.feature_importances_, index=feature_names)

    plt.figure(figsize=(10, 6))
    importances.sort_values(ascending=False).head(top_n).plot(kind="barh")
    plt.title(f"Top {top_n} Feature Importances (Multivariate PULSAR)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def _describe_path_to_leaf(tree, target_leaf, feature_names):
    """Spell out the branch conditions leading from the root to one leaf.

    Args:
        tree: A fitted ``DecisionTreeClassifier``.
        target_leaf: Node id of the leaf to reach.
        feature_names: Names aligned with the tree's features.

    Returns:
        List of condition strings, or None if the leaf is unreachable.
    """
    def walk(node, conditions):
        if node == target_leaf:
            return conditions
        if tree.tree_.children_left[node] == _tree.TREE_LEAF:
            return None

        name = feature_names[tree.tree_.feature[node]]
        threshold = tree.tree_.threshold[node]
        branches = [
            (tree.tree_.children_left[node], f"{name} <= {threshold:.2f}"),
            (tree.tree_.children_right[node], f"{name} > {threshold:.2f}"),
        ]
        for child, condition in branches:
            found = walk(child, conditions + [condition])
            if found is not None:
                return found
        return None

    return walk(0, [])


def lore_explain(forest, train_features, test_features, feature_names):
    """Explain individual predictions with local rules and counter-factuals.

    For each chosen test subject:

    1. Draw ``LORE_N_NEIGHBOURS`` synthetic neighbours around them. The noise is
       scaled by each feature's own standard deviation, so features on wildly
       different scales get proportionate perturbation.
    2. Label the whole cloud with the black-box model.
    3. Fit a shallow decision tree to it and read off the branch the real
       subject follows — that is the local rule.
    4. Walk the tree for the nearest leaf predicting the *other* class — that is
       the counter-factual, the smallest described change that would flip the
       decision.

    If every synthetic neighbour gets the same label, no tree can be fitted and
    that is reported rather than silently skipped.

    Args:
        forest: The fitted black-box classifier being explained.
        train_features: Training matrix, used only for per-feature scale.
        test_features: Test matrix holding the instances to explain.
        feature_names: Names aligned with the feature columns.
    """
    print("\n[LORE] Local rule-based explanations")

    # Per-feature spread of the training data. LORE_EPSILON is added to it
    # below so a constant feature still gets a little noise instead of none.
    feature_stds = np.std(train_features, axis=0)

    for index in LORE_EXPLAIN_INDICES:
        instance = test_features[index].reshape(1, -1)
        predicted_class = int(forest.predict(instance)[0])

        rng = np.random.RandomState(SEEDS["lore"])
        neighbourhood = instance + rng.normal(
            0, (feature_stds + LORE_EPSILON) * LORE_NOISE_SCALE,
            (LORE_N_NEIGHBOURS, instance.shape[1]))
        neighbour_labels = forest.predict(neighbourhood)

        if len(np.unique(neighbour_labels)) <= 1:
            print(f"\ntest idx={index} | ExtraTrees -> "
                  f"{CLASS_NAMES[predicted_class]} | Neighbourhood generated "
                  "only one class, cannot fit LORE tree.")
            continue

        surrogate = DecisionTreeClassifier(max_depth=LORE_TREE_MAX_DEPTH,
                                           random_state=SEEDS["lore"],
                                           class_weight="balanced")
        surrogate.fit(neighbourhood, neighbour_labels)
        fidelity = (surrogate.predict(neighbourhood) == neighbour_labels).mean()

        # --- The rule this instance actually follows ---
        decision_path = surrogate.decision_path(instance)
        leaf_id = surrogate.apply(instance)[0]
        visited_nodes = decision_path.indices[
            decision_path.indptr[0]:decision_path.indptr[1]]

        conditions = []
        for node_id in visited_nodes:
            if node_id == leaf_id:
                continue
            feature_index = surrogate.tree_.feature[node_id]
            threshold = surrogate.tree_.threshold[node_id]
            comparison = ("<=" if instance.flatten()[feature_index] <= threshold
                          else ">")
            conditions.append(f"{feature_names[feature_index]} {comparison} "
                              f"{threshold:.2f}")

        leaf_class = int(np.argmax(surrogate.tree_.value[leaf_id]))
        print(f"\ntest idx={index} | ExtraTrees -> "
              f"{CLASS_NAMES[predicted_class]} | local fidelity={fidelity:.3f}")
        if conditions:
            print(f"Rule (-> class {leaf_class}): IF "
                  + " AND ".join(conditions))
        else:
            print("Rule: (trivial - root is a leaf)")

        # --- The nearest path to the opposite decision ---
        leaves = np.where(
            surrogate.tree_.children_left == _tree.TREE_LEAF)[0]
        for leaf in leaves:
            candidate_class = int(np.argmax(surrogate.tree_.value[leaf]))
            if candidate_class == predicted_class:
                continue
            path = _describe_path_to_leaf(surrogate, leaf, feature_names)
            if path:
                print(f"Counter-factual (-> class {candidate_class}): IF "
                      + " AND ".join(path))
                break
        else:
            print("(no counter-factual leaf at this depth)")
