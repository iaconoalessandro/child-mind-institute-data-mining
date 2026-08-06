"""
Explaining the MLP's predictions (SHAP, LIME, TREPAN, LORE).

A neural network gives no feature importances of its own, so we approximate its
behaviour with four complementary post-hoc methods:

    SHAP    global + local attribution, from cooperative game theory
    LIME    a local linear model fitted around one instance
    TREPAN  a single global decision tree trained to imitate the network
    LORE    a local decision tree, plus a counter-factual ("what would have
            to change for a different prediction?")

All four operate on the original, pre-PCA feature space and explain the whole
fitted pipeline, so the attributions are in units a reader recognises.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import (DecisionTreeClassifier, _tree, export_text,
                          plot_tree)

from src.config import SEEDS

# Test-set rows explained by the local methods, chosen to cover several
# predicted classes.
DEFAULT_INSTANCES = [0, 5, 10, 15]


def shap_explain(pipeline, X_train, X_test, feature_names, class_names,
                 instance_idx=None, n_background=50, n_samples=100):
    """Attribute each prediction to its features using Shapley values.

    ``KernelExplainer`` is model-agnostic: it repeatedly re-runs the pipeline
    with features swapped out for values drawn from a background sample, and
    attributes the change in output to the feature it swapped. That makes it
    slow, hence the small background sample and the 50-row test subset.

    Produces a global summary plot, a mean-|SHAP| bar chart of the top 15
    features, and one force plot per requested instance.

    Args:
        pipeline: Fitted pipeline exposing ``predict`` and ``predict_proba``.
        X_train: Training features as a DataFrame (the background pool).
        X_test: Test features as a DataFrame.
        feature_names: Column names, in order.
        class_names: Display names for the four classes.
        instance_idx: Test rows to explain locally.
        n_background: Background samples given to the explainer.
        n_samples: Perturbations SHAP draws per explanation.
    """
    import shap

    instance_idx = instance_idx or DEFAULT_INSTANCES

    background = shap.sample(X_train, n_background,
                             random_state=SEEDS["random_state"])
    # The explainer hands SHAP a bare numpy array; the pipeline wants named
    # columns, so restore them on the way in.
    explainer = shap.KernelExplainer(
        lambda rows: pipeline.predict_proba(
            pd.DataFrame(rows, columns=feature_names)),
        background,
    )

    X_test_sample = X_test.iloc[:50]
    shap_values = explainer.shap_values(X_test_sample, nsamples=n_samples)

    # SHAP returns a list per class on older versions and a 3D array on newer
    # ones; normalise both to what the two plots below need.
    if isinstance(shap_values, list):
        class_0_values = shap_values[0]
        mean_abs_importance = np.mean(
            [np.abs(values).mean(axis=0) for values in shap_values], axis=0)
    elif shap_values.ndim == 3:
        class_0_values = shap_values[:, :, 0]
        mean_abs_importance = np.abs(shap_values).mean(axis=(0, 2))
    else:
        class_0_values = shap_values
        mean_abs_importance = np.abs(shap_values).mean(axis=0)

    print("\n[SHAP] Global feature importance — class sii=0")
    shap.summary_plot(class_0_values, X_test_sample,
                      feature_names=feature_names, show=True)

    print("\n[SHAP] Mean |SHAP| across all classes")
    ranking = np.argsort(mean_abs_importance)[::-1]
    top_15 = ranking[:15]
    plt.figure(figsize=(8, 6))
    # Reversed so the most important feature ends up at the top of the chart.
    plt.barh([feature_names[i] for i in top_15][::-1],
             mean_abs_importance[top_15][::-1])
    plt.xlabel("mean(|SHAP value|)")
    plt.title("SHAP global importance (top 15 features)")
    plt.tight_layout()
    plt.show()

    for index in instance_idx:
        instance = X_test.iloc[[index]]
        local_values = explainer.shap_values(instance, nsamples=n_samples)
        predicted_class = int(pipeline.predict(instance)[0])

        print(f"\n[SHAP] test idx={index}, "
              f"predicted {class_names[predicted_class]}")

        if isinstance(local_values, list):
            contributions = local_values[predicted_class][0]
        elif local_values.ndim == 3:
            contributions = local_values[0, :, predicted_class]
        else:
            contributions = local_values[0]

        shap.force_plot(
            explainer.expected_value[predicted_class],
            contributions,
            instance.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        figure, axis = plt.gcf(), plt.gca()
        figure.set_size_inches(16, 6)
        # Force-plot labels overlap badly at this width; angle them.
        for text in axis.texts:
            if "=" in text.get_text():
                text.set_rotation(-45)
                text.set_ha("left")
                text.set_va("top")
        plt.show()
        plt.close(figure)


def lime_explain(pipeline, X_train, X_test, feature_names, class_names,
                 instance_idx=None):
    """Explain single predictions with a locally-fitted linear model.

    LIME perturbs one instance, asks the pipeline to score each perturbation,
    and fits a weighted linear model to those answers. The coefficients say
    which features pushed *this* prediction which way — a local story, not a
    global one.

    Args:
        pipeline: Fitted pipeline exposing ``predict`` and ``predict_proba``.
        X_train: Training features as a DataFrame (defines feature ranges).
        X_test: Test features as a DataFrame.
        feature_names: Column names, in order.
        class_names: Display names for the four classes.
        instance_idx: Test rows to explain.
    """
    from lime.lime_tabular import LimeTabularExplainer

    instance_idx = instance_idx or DEFAULT_INSTANCES

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=SEEDS["random_state"],
    )

    for index in instance_idx:
        row = X_test.iloc[index].values
        predicted_class = int(pipeline.predict(row.reshape(1, -1))[0])
        explanation = explainer.explain_instance(
            data_row=row,
            predict_fn=pipeline.predict_proba,
            num_features=10,
            top_labels=1,
        )

        print(f"\n[LIME] test idx={index} — "
              f"predicted {class_names[predicted_class]}")
        for feature, weight in explanation.as_list(label=predicted_class):
            sign = "+" if weight >= 0 else "-"
            print(f"  {sign} {feature}: {weight:+.4f}")

        explanation.as_pyplot_figure(label=predicted_class)
        plt.title(f"LIME — test idx={index} "
                  f"(predicted {class_names[predicted_class]})")
        plt.show()


def trepan_explain(oracle, X_train, X_test, y_test, feature_names,
                   max_depth=5, n_synth=2000):
    """Approximate the whole network with one readable decision tree.

    The trick is that the network can label as much data as we like. We
    generate synthetic rows by sampling each feature independently from its
    training distribution, ask the network what it would predict for them, and
    fit a shallow tree to those answers. The tree is judged on *fidelity* —
    how often it agrees with the network — not on accuracy against the truth.

    Args:
        oracle: The fitted pipeline being explained.
        X_train: Training features, used as the sampling distribution.
        X_test: Test features, used to measure fidelity.
        y_test: True test labels, used to report the surrogate's own accuracy.
        feature_names: Column names, in order.
        max_depth: Depth of the surrogate tree.
        n_synth: Number of synthetic rows to generate.
    """
    rng = np.random.RandomState(SEEDS["surrogate_tree"])
    # Sample each column independently, which deliberately breaks the
    # correlations between features and probes the network off-distribution.
    synthetic = np.vstack([
        rng.choice(X_train[column].values, size=n_synth, replace=True)
        for column in feature_names
    ]).T

    queries = pd.concat(
        [X_train, pd.DataFrame(synthetic, columns=feature_names)],
        ignore_index=True,
    )
    oracle_labels = oracle.predict(queries)

    surrogate = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=SEEDS["surrogate_tree"],
        class_weight="balanced",
    )
    surrogate.fit(queries, oracle_labels)

    oracle_test = oracle.predict(X_test)
    surrogate_test = surrogate.predict(X_test)
    print(f"\n[TREPAN] Surrogate fidelity to MLP on test set: "
          f"{(oracle_test == surrogate_test).mean():.4f}")
    print(f"[TREPAN] Surrogate accuracy on true y_test:   "
          f"{(surrogate_test == y_test).mean():.4f}")

    print("\n[TREPAN] Top-level rules:")
    print(export_text(surrogate, feature_names=feature_names, max_depth=3))

    plt.figure(figsize=(22, 10))
    plot_tree(surrogate, feature_names=feature_names,
              class_names=[str(c) for c in surrogate.classes_],
              filled=True, max_depth=3, fontsize=8)
    plt.title("TREPAN — Decision-tree surrogate (top 3 levels)")
    plt.tight_layout()
    plt.show()


def _describe_path_to(tree, target_node, feature_names):
    """Return the list of split conditions leading from the root to a node.

    Args:
        tree: A fitted ``DecisionTreeClassifier``.
        target_node: Index of the node to reach.
        feature_names: Column names, in order.

    Returns:
        List of condition strings, or None if the node is unreachable.
    """
    def walk(node, conditions):
        if node == target_node:
            return conditions
        if tree.tree_.children_left[node] == _tree.TREE_LEAF:
            return None
        feature = feature_names[tree.tree_.feature[node]]
        threshold = tree.tree_.threshold[node]
        return (walk(tree.tree_.children_left[node],
                     conditions + [f"{feature} <= {threshold:.2f}"])
                or walk(tree.tree_.children_right[node],
                        conditions + [f"{feature} > {threshold:.2f}"]))

    return walk(0, [])


def lore_explain(oracle, X_train, X_test, feature_names, class_names,
                 instance_idx=None):
    """Explain single predictions as an IF-THEN rule plus a counter-factual.

    For each instance we build a cloud of nearby points, label them with the
    network, and fit a shallow tree to that neighbourhood. The branch the
    instance falls down becomes the rule; the nearest leaf with a *different*
    class becomes the counter-factual — what would have to change for the
    prediction to flip.

    The neighbourhood is drawn from NumPy's global random generator, so the
    rules shift slightly between runs. ``X_train`` is accepted for interface
    consistency with the other explainers but is not used here.

    Args:
        oracle: The fitted pipeline being explained.
        X_train: Training features (unused; see above).
        X_test: Test features as a DataFrame.
        feature_names: Column names, in order.
        class_names: Display names for the four classes.
        instance_idx: Test rows to explain.
    """
    instance_idx = instance_idx or DEFAULT_INSTANCES
    print("\n[LORE] Local rule-based explanations")

    for index in instance_idx:
        instance = X_test.iloc[[index]].values.flatten().astype(float)
        predicted_class = int(oracle.predict(instance.reshape(1, -1))[0])

        neighbourhood = instance.reshape(1, -1) + np.random.normal(
            0, 0.5, (800, len(instance))
        )
        neighbour_labels = oracle.predict(neighbourhood)

        local_tree = DecisionTreeClassifier(
            max_depth=4, random_state=SEEDS["surrogate_tree"],
            class_weight="balanced",
        )
        local_tree.fit(neighbourhood, neighbour_labels)
        local_fidelity = (local_tree.predict(neighbourhood)
                          == neighbour_labels).mean()

        # Walk the branch this instance actually takes to recover its rule.
        decision_path = local_tree.decision_path(instance.reshape(1, -1))
        leaf_id = local_tree.apply(instance.reshape(1, -1))[0]
        split_feature = local_tree.tree_.feature
        split_threshold = local_tree.tree_.threshold

        conditions = []
        for node in decision_path.indices[decision_path.indptr[0]:
                                          decision_path.indptr[1]]:
            if node == leaf_id:
                continue  # the leaf itself carries no split condition
            comparison = ("<=" if instance[split_feature[node]]
                          <= split_threshold[node] else ">")
            conditions.append(f"{feature_names[split_feature[node]]} "
                              f"{comparison} {split_threshold[node]:.2f}")

        leaf_class = int(np.argmax(local_tree.tree_.value[leaf_id]))
        print(f"\n  test idx={index} | MLP → {class_names[predicted_class]} "
              f"| local fidelity={local_fidelity:.3f}")
        if conditions:
            print(f"  Rule (→ class {leaf_class}): IF "
                  + " AND ".join(conditions))
        else:
            print("  Rule: (trivial — root is a leaf)")

        # Counter-factual: the first leaf predicting a different class.
        leaves = np.where(local_tree.tree_.children_left == _tree.TREE_LEAF)[0]
        counter_factual = None
        for leaf in leaves:
            if int(np.argmax(local_tree.tree_.value[leaf])) == predicted_class:
                continue
            path = _describe_path_to(local_tree, leaf, feature_names)
            if path:
                counter_factual = (
                    int(np.argmax(local_tree.tree_.value[leaf])), path)
                break

        if counter_factual:
            leaf_class, path = counter_factual
            print(f"  Counter-factual (→ class {leaf_class}): IF "
                  + " AND ".join(path))
        else:
            print("  (no counter-factual leaf at this depth)")
