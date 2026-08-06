"""
Choosing an imputation strategy (part of STEP 3, preprocessing).

Roughly a third of the cells in this dataset are missing, so *how* we fill them
in matters. To compare strategies fairly we need ground truth, which we
manufacture: hide 10% of the cells that already hold a real value, let each
candidate imputer reconstruct them from the rest, and compare its guesses to
the values we hid.

The regression and classification tracks score candidates differently and
therefore search slightly different spaces — see the two objective functions.

A known caveat, kept deliberately
---------------------------------
The imputer is selected once, on the whole training pool, and only afterwards
re-fitted inside each cross-validation fold. So the *choice* of imputer has
seen rows that later act as validation data. The bias is small — imputation
here never looks at the target y, and every fold still re-fits from scratch —
but the CV scores should be read as mildly optimistic. Fully nested selection
was judged not worth the compute.
"""

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, Lasso, LogisticRegression, Ridge

from src.config import IMPUTER_MASK_FRACTION, SEEDS

# Score returned when an imputer errors out, so the trial is ranked last.
# The sign flips with the study direction: regression maximises, classification
# minimises.
FAILED_TRIAL_SCORE_MAXIMISE = -1e9
FAILED_TRIAL_SCORE_MINIMISE = 1e9


def create_mask(X, p=IMPUTER_MASK_FRACTION, random_state=None):
    """Hide a fraction of the cells that currently hold a real value.

    Works column by column so that every feature contributes evaluation cells
    in proportion to how complete it already is.

    Args:
        X: 2D numpy array, may contain NaNs.
        p: Fraction of each column's observed cells to hide.
        random_state: Seed; defaults to ``SEEDS["mask_rng"]``.

    Returns:
        Boolean array of X's shape, True where a value should be hidden.
    """
    rng = np.random.RandomState(
        SEEDS["mask_rng"] if random_state is None else random_state
    )
    mask = np.zeros(X.shape, dtype=bool)
    # Deliberately a per-column loop: it fixes the order in which the random
    # generator is consumed, which is what makes the mask reproducible.
    for column in range(X.shape[1]):
        observed = np.where(~np.isnan(X[:, column]))[0]
        n_to_hide = int(len(observed) * p)
        if n_to_hide > 0:
            mask[rng.choice(observed, n_to_hide, replace=False), column] = True
    return mask


def evaluate_imputation(X_true, X_imputed, mask, y=None):
    """Score reconstructed values against the ones that were hidden.

    Four complementary views, computed per column and then averaged:

    * **spearman** — does the imputer preserve the ranking of values?
    * **wasserstein** — does the imputed distribution match the real one?
    * **var_ratio** — does it preserve spread, or regress everything to the
      mean? (1.0 is ideal; below 1 means over-smoothing.)
    * **nrmse** — root mean squared error divided by the column's range, so
      columns on different scales stay comparable.

    Args:
        X_true: Original matrix holding the real values.
        X_imputed: Matrix after imputation.
        mask: Boolean array marking the hidden cells.
        y: Optional labels; enables a per-class variance-ratio diagnostic that
            catches imputers which flatten differences between severity groups.

    Returns:
        Dict with keys ``spearman``, ``wasserstein``, ``var_ratio``,
        ``class_var_ratios`` and ``nrmse``, each averaged across columns.
    """
    per_column = {"spearman": [], "wasserstein": [], "var_ratio": [],
                  "class_var_ratios": [], "nrmse": []}
    classes = np.unique(y) if y is not None else []

    for column in range(X_true.shape[1]):
        hidden = mask[:, column]
        if not hidden.any():
            continue
        actual, predicted = X_true[hidden, column], X_imputed[hidden, column]

        # Spearman is undefined if either side is constant; score those 0.
        if len(np.unique(actual)) > 1 and len(np.unique(predicted)) > 1:
            rho, _ = spearmanr(actual, predicted)
            per_column["spearman"].append(rho if np.isfinite(rho) else 0)
        else:
            per_column["spearman"].append(0)

        per_column["wasserstein"].append(wasserstein_distance(actual, predicted))

        actual_variance = np.var(actual)
        per_column["var_ratio"].append(
            np.var(predicted) / actual_variance if actual_variance > 0 else 1.0
        )

        value_range = np.ptp(actual)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        per_column["nrmse"].append(rmse / value_range if value_range > 0 else 0.0)

        if y is not None:
            hidden_labels = y[hidden]
            for class_label in classes:
                in_class = hidden_labels == class_label
                # Need enough cells for a variance estimate to mean anything.
                if in_class.sum() > 5:
                    class_variance = np.var(actual[in_class])
                    if class_variance > 0:
                        per_column["class_var_ratios"].append(
                            np.var(predicted[in_class]) / class_variance
                        )

    # Average each metric; a metric with no samples defaults to its neutral
    # value (1.0 for ratios, 0.0 for errors and correlations).
    averaged = {
        name: (np.mean(values) if values else (1.0 if "ratio" in name else 0.0))
        for name, values in per_column.items()
    }
    if not per_column["class_var_ratios"]:
        averaged["class_var_ratios"] = averaged["var_ratio"]
    return averaged


def _score_candidate(imputer, X_raw, mask, y_labels):
    """Fit ``imputer`` on the masked matrix and score its reconstruction.

    Args:
        imputer: Candidate imputer, or None if the trial picked an
            unimplemented option.
        X_raw: Original feature matrix.
        mask: Cells to hide before fitting.
        y_labels: Optional labels for the per-class diagnostic.

    Returns:
        The metric dict from :func:`evaluate_imputation`.
    """
    masked = X_raw.copy()
    masked[mask] = np.nan
    return evaluate_imputation(X_raw, imputer.fit_transform(masked), mask,
                               y=y_labels)


def imputation_objective_regression(trial, X_raw, mask, y_labels=None):
    """Optuna objective for the regression track — **maximise**.

    Scores ``0.7 * spearman - 0.3 * nrmse``: mostly rewarding an imputer that
    preserves the ordering of values (which is what an ordinal target cares
    about), with a smaller penalty for absolute reconstruction error.

    Note that ``"knn"`` appears in the choice list but has no branch below, so
    those trials raise and are scored ``-1e9``. That is a defect in the original
    study, kept unchanged because removing it would alter which
    hyperparameters the sampler explores and therefore the published result.

    Args:
        trial: Optuna trial.
        X_raw: Training feature matrix as a numpy array.
        mask: Boolean mask from :func:`create_mask`.
        y_labels: Optional label array for the per-class diagnostic.

    Returns:
        Float score to maximise.
    """
    choice = trial.suggest_categorical(
        "imputer_choice",
        ["knn", "lasso", "ridge", "bayesian_ridge", "mice", "extra_trees"],
    )
    seed = SEEDS["random_state"]
    imputer = None

    if choice == "mice":
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=trial.suggest_int("mice_et_estimators", 5, 20),
                max_depth=trial.suggest_int("mice_et_max_depth", 2, 5),
                random_state=seed, n_jobs=-1,
            ),
            max_iter=trial.suggest_int("mice_max_iter", 5, 15),
            initial_strategy=trial.suggest_categorical(
                "mice_initial_strategy", ["mean", "median"]),
            random_state=seed,
        )
    elif choice == "extra_trees":
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=trial.suggest_int("et_estimators", 5, 20),
                max_depth=trial.suggest_int("et_max_depth", 2, 5),
                random_state=seed, n_jobs=-1,
            ),
            max_iter=trial.suggest_int("et_max_iter", 3, 10),
            initial_strategy=trial.suggest_categorical(
                "et_initial_strategy", ["mean", "median"]),
            random_state=seed,
        )
    elif choice == "lasso":
        imputer = IterativeImputer(
            estimator=Lasso(
                alpha=trial.suggest_float("alpha", 0.001, 0.5, log=True),
                random_state=seed),
            random_state=seed,
        )
    elif choice == "ridge":
        imputer = IterativeImputer(
            estimator=Ridge(
                alpha=trial.suggest_float("alpha_ridge", 0.001, 100.0, log=True),
                random_state=seed),
            random_state=seed,
        )
    elif choice == "bayesian_ridge":
        imputer = IterativeImputer(
            estimator=BayesianRidge(
                alpha_1=trial.suggest_float("br_alpha_1", 1e-7, 1e-3, log=True),
                alpha_2=trial.suggest_float("br_alpha_2", 1e-7, 1e-3, log=True),
                lambda_1=trial.suggest_float("br_lambda_1", 1e-7, 1e-3, log=True),
                lambda_2=trial.suggest_float("br_lambda_2", 1e-7, 1e-3, log=True)),
            random_state=seed,
        )

    try:
        scores = _score_candidate(imputer, X_raw, mask, y_labels)
    except Exception:
        return FAILED_TRIAL_SCORE_MAXIMISE
    return 0.7 * scores["spearman"] - 0.3 * scores["nrmse"]


def imputation_objective_classification(trial, X_raw, mask, y_labels=None):
    """Optuna objective for the classification track — **minimise**.

    Scores plain NRMSE: for a nominal target we care about reconstructing
    values accurately rather than preserving their ranking.

    The search space is wider than the regression one: it adds ``knn`` and
    ``logistic``, and gives the tree-based imputers more estimators and depth.

    Args:
        trial: Optuna trial.
        X_raw: Training feature matrix as a numpy array.
        mask: Boolean mask from :func:`create_mask`.
        y_labels: Optional label array for the per-class diagnostic.

    Returns:
        NRMSE to minimise.
    """
    choice = trial.suggest_categorical(
        "imputer_choice",
        ["knn", "lasso", "ridge", "logistic", "bayesian_ridge", "mice",
         "extra_trees"],
    )
    seed = SEEDS["random_state"]
    imputer = None

    if choice == "knn":
        imputer = KNNImputer(n_neighbors=trial.suggest_int("n_neighbors", 3, 30))
    elif choice == "mice":
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=trial.suggest_int("mice_et_estimators", 10, 50),
                max_depth=trial.suggest_int("mice_et_max_depth", 3, 8),
                random_state=seed, n_jobs=-1,
            ),
            max_iter=trial.suggest_int("mice_max_iter", 5, 15),
            initial_strategy=trial.suggest_categorical(
                "mice_initial_strategy", ["mean", "median"]),
            random_state=seed,
        )
    elif choice == "extra_trees":
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=trial.suggest_int("et_estimators", 10, 40),
                max_depth=trial.suggest_int("et_max_depth", 3, 8),
                random_state=seed, n_jobs=-1,
            ),
            max_iter=trial.suggest_int("et_max_iter", 3, 10),
            initial_strategy=trial.suggest_categorical(
                "et_initial_strategy", ["mean", "median"]),
            random_state=seed,
        )
    elif choice == "lasso":
        imputer = IterativeImputer(
            estimator=Lasso(
                alpha=trial.suggest_float("alpha", 0.001, 0.5, log=True),
                random_state=seed),
            random_state=seed,
        )
    elif choice == "ridge":
        imputer = IterativeImputer(
            estimator=Ridge(
                alpha=trial.suggest_float("alpha_ridge", 0.001, 100.0, log=True),
                random_state=seed),
            random_state=seed,
        )
    elif choice == "logistic":
        imputer = IterativeImputer(
            estimator=LogisticRegression(
                C=trial.suggest_float("C_logistic", 0.01, 10.0, log=True),
                random_state=seed, max_iter=1000),
            random_state=seed,
        )
    elif choice == "bayesian_ridge":
        imputer = IterativeImputer(
            estimator=BayesianRidge(
                alpha_1=trial.suggest_float("br_alpha_1", 1e-7, 1e-3, log=True),
                alpha_2=trial.suggest_float("br_alpha_2", 1e-7, 1e-3, log=True),
                lambda_1=trial.suggest_float("br_lambda_1", 1e-7, 1e-3, log=True),
                lambda_2=trial.suggest_float("br_lambda_2", 1e-7, 1e-3, log=True)),
            random_state=seed,
        )

    try:
        scores = _score_candidate(imputer, X_raw, mask, y_labels)
    except Exception:
        return FAILED_TRIAL_SCORE_MINIMISE
    return scores["nrmse"]
