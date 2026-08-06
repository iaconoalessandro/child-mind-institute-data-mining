"""
Single entry point for the whole pipeline.

Run a stage by name:

    python main.py --stage prep-reg      # build the regression dataset
    python main.py --stage tune-reg      # tune the 8 regression models
    python main.py --stage eval-reg      # final regression evaluation
    python main.py --stage prep-clf
    python main.py --stage tune-clf
    python main.py --stage eval-clf
    python main.py --stage all           # everything, in order

The stages are ordered and depend on each other: preprocessing writes the CSVs
that tuning reads, and tuning writes the trial files that evaluation reads.

Tuning is slow — the full sweep is many hours of CPU. Use ``--models`` to tune
a subset and ``--trials`` to shorten each study while you are experimenting.

Plots are shown interactively. When running headless, set a non-interactive
matplotlib backend first: ``MPLBACKEND=Agg python main.py --stage eval-reg``.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from src.config import (IMPUTER_MASK_FRACTION, IMPUTER_PARAMS_CLF,
                        IMPUTER_PARAMS_REG, IMPUTER_PARAMS_REG_EVAL,
                        IMPUTER_STUDY_N_JOBS, IMPUTER_STUDY_TRIALS,
                        OUTLIER_PERCENTILE, PATHS, SII_CLASS_NAMES,
                        STUDY_TRIALS)
from src.data_loader import (DROP_FOR_CLASSIFICATION, DROP_FOR_REG_EVAL,
                             DROP_FOR_TUNING, load_processed_split,
                             preprocess_split)

# Heavier dependencies are imported inside the stage that needs them, so that
# running one stage does not require every library the project can use — the
# preprocessing stages, for instance, need no gradient-boosting library.

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# STEP 2 & 3 — Preprocessing
# ---------------------------------------------------------------------------

def run_preprocessing(track):
    """Build one of the two processed datasets from the raw CSV.

    Runs the full chain: audit -> drop leaking columns -> stratified split ->
    train-only feature filters -> imputer search -> outlier detection -> save.

    The two tracks differ only in how they score candidate imputers: regression
    rewards rank preservation, classification rewards raw accuracy.

    Args:
        track: ``"reg"`` or ``"clf"``.
    """
    import optuna

    from src.data_loader import (apply_train_only_filters, audit_data,
                                 drop_leakage_columns, load_raw_dataset,
                                 save_processed_split, split_train_test,
                                 validate_processed_file)
    from src.imputer_search import (create_mask,
                                    imputation_objective_classification,
                                    imputation_objective_regression)
    from src.outliers import detect_consensus_outliers, summarise_outliers

    is_regression = track == "reg"
    label = "Regression" if is_regression else "Classification"
    output_path = PATHS["reg_processed" if is_regression else "clf_processed"]
    os.makedirs(PATHS["results_dir"], exist_ok=True)

    print(f"\n{'=' * 70}\n{label} preprocessing\n{'=' * 70}")

    dataset = audit_data(load_raw_dataset())
    print(f"Dataset shape after audit: {dataset.shape}")

    numeric_features = drop_leakage_columns(dataset)
    print(f"Shape after leakage-column drop: {numeric_features.shape}")
    print("NOTE: >50% missing and Spearman-correlation filters are applied "
          "AFTER the split,\n      computed on the training rows only, so no "
          "test row decides which columns survive.")

    X_train, X_test, y_train, y_test = split_train_test(numeric_features)
    print(f"\nLabelled pool (non-null sii): {len(X_train) + len(X_test)} rows")
    print(f"X_train: {X_train.shape}\nX_test:  {X_test.shape}")

    X_train, X_test = apply_train_only_filters(X_train, X_test, label=label)

    # --- Imputer search ---------------------------------------------------
    # Hide 10% of the known values, then let Optuna find the imputer that
    # reconstructs them best.
    benchmark_matrix = X_train.values.copy()
    mask = create_mask(benchmark_matrix, p=IMPUTER_MASK_FRACTION)
    print(f"\nBenchmark matrix: {benchmark_matrix.shape}")
    print(f"Masked entries:   {mask.sum()} "
          f"({100 * mask.sum() / np.prod(benchmark_matrix.shape):.1f}% of cells)\n")

    objective = (imputation_objective_regression if is_regression
                 else imputation_objective_classification)
    direction = "maximize" if is_regression else "minimize"

    print(f"=== {label} Imputer Study ({IMPUTER_STUDY_TRIALS} trials) ===")
    study = optuna.create_study(direction=direction)
    study.optimize(
        lambda trial: objective(trial, benchmark_matrix, mask,
                                y_labels=y_train.values),
        n_trials=IMPUTER_STUDY_TRIALS,
        show_progress_bar=True,
        n_jobs=IMPUTER_STUDY_N_JOBS,
    )
    print(f"Best: {study.best_params}  |  score={study.best_value:.4f}\n")
    print("The winning configuration is recorded in src/config.py as "
          f"IMPUTER_PARAMS_{track.upper()} and is what downstream stages use.")

    # --- Outlier detection ------------------------------------------------
    outlier_mask = summarise_outliers(
        detect_consensus_outliers(X_train, OUTLIER_PERCENTILE), y_train.values
    )

    # --- Save -------------------------------------------------------------
    save_processed_split(X_train, y_train, X_test, y_test, outlier_mask,
                         output_path)
    validate_processed_file(output_path)
    print(f"\n{os.path.basename(output_path)} saved. Done.")


# ---------------------------------------------------------------------------
# STEP 4 — Tuning
# ---------------------------------------------------------------------------

def run_tuning(track, model_keys=None, n_trials=None):
    """Tune every model in a track with Optuna and save the trial history.

    Args:
        track: ``"reg"`` or ``"clf"``.
        model_keys: Subset of model keys to tune; defaults to all of them.
        n_trials: Override the per-study trial budget from ``config``.
    """
    from src.tuning import (CLASSIFICATION_MODELS, REGRESSION_MODELS,
                            tune_model)

    is_regression = track == "reg"
    registry = REGRESSION_MODELS if is_regression else CLASSIFICATION_MODELS
    imputer_params = IMPUTER_PARAMS_REG if is_regression else IMPUTER_PARAMS_CLF
    processed_path = PATHS["reg_processed" if is_regression else "clf_processed"]
    drop_columns = DROP_FOR_TUNING if is_regression else DROP_FOR_CLASSIFICATION

    os.makedirs(PATHS["results_dir"], exist_ok=True)
    X_train, y_train, _, _ = load_processed_split(processed_path, drop_columns)

    print(f"X_train shape: {X_train.shape} "
          f"(NaNs present: {X_train.isnull().any().any()})")
    print(f"Active imputer configuration: {imputer_params}")

    for model_key in (model_keys or registry):
        budget = n_trials or STUDY_TRIALS[
            "regression" if is_regression else "classification"][model_key]
        print(f"\n--- {model_key} ({budget} trials) ---")
        tune_model(model_key, track, X_train, y_train, imputer_params,
                   PATHS["results_dir"], n_trials=budget)


# ---------------------------------------------------------------------------
# STEP 5 — Evaluation
# ---------------------------------------------------------------------------

def run_regression_evaluation():
    """Refit the tuned regressors on the full training set and score on test.

    Imports are local because this stage pulls in the gradient-boosting
    libraries, which the preprocessing and pattern-mining stages do not need.
    """
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler

    from src.config import EVAL_FIXED_PARAMS_REG
    from src.data_loader import build_imputer
    from src.metrics import (build_regression_stack, compare_regressor_results,
                             interpret_meta_weights, plot_confusion_matrix,
                             train_and_evaluate_regressors)
    from src.models import TorchRegressor, build_model
    from src.utils import load_tuned_params_by_prefix

    print("Loading data (raw features)...")
    X_train_raw, y_train, X_test_raw, y_test = load_processed_split(
        PATHS["reg_processed"], DROP_FOR_REG_EVAL
    )
    print(f"X_train_raw: {X_train_raw.shape} "
          f"(has NaNs: {X_train_raw.isnull().any().any()})")
    print(f"Test sii distribution: {dict(y_test.value_counts())}\n")

    # Note the order here: impute first, then scale. This stage deliberately
    # differs from the fold pipeline used during tuning (which scales first),
    # and uses the higher-scoring extra_trees imputer rather than the mice one.
    print("Fitting final imputer on the full training set...")
    imputer = build_imputer(IMPUTER_PARAMS_REG_EVAL)
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imputed)
    X_test = scaler.transform(X_test_imputed)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    # Model name -> (estimator class, filename fragment of its trials CSV).
    model_specs = {
        "LightGBM": (lgb.LGBMRegressor, "lgb"),
        "XGBoost": (xgb.XGBRegressor, "xgb"),
        "CatBoost": (cb.CatBoostRegressor, "cb"),
        "RandomForest": (RandomForestRegressor, "rf"),
        "Ridge": (Ridge, "ridge"),
        "Lasso": (Lasso, "lasso"),
    }
    tuned = {name: load_tuned_params_by_prefix(prefix, PATHS["results_dir"])
             for name, (_, prefix) in model_specs.items()}

    mlp_params, mlp_pca_n = load_tuned_params_by_prefix("nn", PATHS["results_dir"])
    print(f"Hyperparameters loaded for: {list(tuned)}"
          f"{' + MLP' if mlp_params else ''}")

    def make(name):
        """Build one estimator from its tuned params and fixed kwargs."""
        model_class = model_specs[name][0]
        params, pca_n = tuned[name]
        return build_model(model_class, params, pca_n, EVAL_FIXED_PARAMS_REG[name])

    models = {name: make(name) for name in model_specs}
    if mlp_params:
        models["MLP"] = build_model(
            TorchRegressor, {**mlp_params, "device": "cpu", "verbose": False},
            mlp_pca_n,
        )

    results = train_and_evaluate_regressors(X_train, y_train, X_test, y_test,
                                            models)
    leaderboard = compare_regressor_results(results)

    best_name = leaderboard.iloc[0]["Model"]
    best_predictions = results[best_name]["Test_Class_Preds"]
    plot_confusion_matrix(y_test, best_predictions,
                          f"Test Set Confusion Matrix - {best_name}")
    print(f"Classification Report ({best_name}):\n")
    print(classification_report(y_test, best_predictions))

    # The MLP is left out of the stack: its test QWK (~0.17) sits far below the
    # tree models (0.35+), and including it dragged the ensemble down.
    print("Constructing Level-1 Stacking Regressor...")
    stack_estimators = [(name, make(name)) for name in model_specs]
    stacking_model, _ = build_regression_stack(X_train, y_train, X_test,
                                               y_test, stack_estimators)
    interpret_meta_weights(stacking_model)


def run_classification_evaluation(explain=True):
    """Refit the tuned classifiers, evaluate them, explain the MLP, then stack.

    Args:
        explain: Run the four explainability methods. They are slow (SHAP's
            KernelExplainer dominates), so they can be skipped.
    """
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.preprocessing import label_binarize
    from sklearn.svm import SVC
    from sklearn.utils import class_weight

    from src.config import (EVAL_FIXED_PARAMS_CLF, SII_CLASSES,
                            STACK_FIXED_PARAMS_CLF)
    from src.metrics import build_classification_stack, evaluate_classifier
    from src.models import TorchClassifier, build_model
    from src.utils import load_tuned_params

    X_train_raw, y_train, X_test_raw, y_test = load_processed_split(
        PATHS["clf_processed"], DROP_FOR_CLASSIFICATION
    )

    # Impute and scale once; every classifier below trains on this matrix.
    X_train_array, X_test_array, _, _ = preprocess_split(
        X_train_raw, X_test_raw, y_train, IMPUTER_PARAMS_CLF
    )
    X_train = pd.DataFrame(X_train_array, columns=X_train_raw.columns)
    X_test = pd.DataFrame(X_test_array, columns=X_train_raw.columns)
    y_test_binary = label_binarize(y_test, classes=SII_CLASSES)
    print(f"Final X_train shape: {X_train.shape}")
    print(f"Final X_test shape:  {X_test.shape}")

    def load(name):
        """Load one model's tuned params from optuna trials/<name>_clf_trials.csv."""
        return load_tuned_params(
            os.path.join(PATHS["results_dir"], f"{name}_clf_trials.csv"))

    # Display name -> (estimator class, trials-CSV stem).
    model_specs = {
        "LightGBM": (lgb.LGBMClassifier, "lgbm"),
        "XGBoost": (xgb.XGBClassifier, "xgb"),
        "CatBoost": (cb.CatBoostClassifier, "cb"),
        "RandomForest": (RandomForestClassifier, "rf"),
        "RidgeClassifier": (RidgeClassifier, "ridge"),
        "LogisticRegression": (LogisticRegression, "lr"),
        "SVC": (SVC, "svc"),
    }

    fitted = {}
    for name, (model_class, stem) in model_specs.items():
        params, pca_n = load(stem)
        model = build_model(model_class, params, pca_n,
                            EVAL_FIXED_PARAMS_CLF[name])

        if name == "XGBoost":
            # XGBoost has no class_weight argument, so imbalance is handled by
            # per-sample weights passed through the pipeline to the estimator.
            model.fit(X_train, y_train, model__sample_weight=
                      class_weight.compute_sample_weight("balanced", y_train))
        else:
            model.fit(X_train, y_train)

        if name == "RidgeClassifier":
            # RidgeClassifier has no predict_proba. Convert its decision
            # scores to pseudo-probabilities with a numerically stable softmax
            # so the ROC/AUC plots still work.
            scores = model.decision_function(X_test)
            exponentiated = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        else:
            probabilities = model.predict_proba(X_test)

        evaluate_classifier(name, model, X_test, y_test, probabilities,
                            y_test_binary=y_test_binary)
        fitted[name] = model

    # --- MLP --------------------------------------------------------------
    mlp_params, mlp_pca_n = load("nn")
    mlp = build_model(TorchClassifier, mlp_params, mlp_pca_n,
                      EVAL_FIXED_PARAMS_CLF["MLP"])
    # No sample weights needed: TorchClassifier computes balanced class
    # weights internally and folds them into its loss function.
    mlp.fit(X_train, y_train)
    evaluate_classifier("TorchClassifier", mlp, X_test, y_test,
                        mlp.predict_proba(X_test), y_test_binary=y_test_binary)

    if explain:
        run_explainability(mlp, X_train, X_test, y_test)

    # --- Stacking ---------------------------------------------------------
    # Four complementary base learners: two tree ensembles, a kernel method and
    # the network. Each is rebuilt fresh so the stacker fits its own copies.
    print("\n[Stacking Ensemble Evaluation - Threshold Optimized for Macro F1]")
    stack_specs = {
        "rf": (RandomForestClassifier, "rf"),
        "xgb": (xgb.XGBClassifier, "xgb"),
        "svc": (SVC, "svc"),
        "nn": (TorchClassifier, "nn"),
    }
    stack_estimators = []
    for key, (model_class, stem) in stack_specs.items():
        params, pca_n = load(stem)
        stack_estimators.append(
            (key, build_model(model_class, params, pca_n,
                              STACK_FIXED_PARAMS_CLF[key])))

    build_classification_stack(X_train, y_train, X_test, y_test,
                               stack_estimators, class_names=SII_CLASS_NAMES)


def run_explainability(mlp_pipeline, X_train, X_test, y_test):
    """Run SHAP, LIME, TREPAN and LORE against the fitted MLP pipeline.

    Args:
        mlp_pipeline: The fitted MLP pipeline to explain.
        X_train: Imputed and scaled training features (pre-PCA).
        X_test: Imputed and scaled test features (pre-PCA).
        y_test: True test labels, used to score the TREPAN surrogate.
    """
    from src.explainability import (lime_explain, lore_explain, shap_explain,
                                    trepan_explain)

    feature_names = X_train.columns.tolist()
    shap_explain(mlp_pipeline, X_train, X_test, feature_names, SII_CLASS_NAMES)
    lime_explain(mlp_pipeline, X_train, X_test, feature_names, SII_CLASS_NAMES)
    trepan_explain(mlp_pipeline, X_train, X_test, y_test, feature_names)
    lore_explain(mlp_pipeline, X_train, X_test, feature_names, SII_CLASS_NAMES)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

STAGES = ["prep-reg", "tune-reg", "eval-reg",
          "prep-clf", "tune-clf", "eval-clf"]


def main():
    """Parse arguments and run the requested stage or stages."""
    parser = argparse.ArgumentParser(
        description="Run the CMI internet-use severity pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", required=True, choices=STAGES + ["all"],
                        help="Which pipeline stage to run.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Tuning stages only: limit to these model keys "
                             "(e.g. --models lgbm rf).")
    parser.add_argument("--trials", type=int, default=None,
                        help="Tuning stages only: override the trial budget.")
    parser.add_argument("--no-explain", action="store_true",
                        help="eval-clf only: skip SHAP/LIME/TREPAN/LORE.")
    args = parser.parse_args()

    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n{'#' * 70}\n# STAGE: {stage}\n{'#' * 70}")
        if stage == "prep-reg":
            run_preprocessing("reg")
        elif stage == "prep-clf":
            run_preprocessing("clf")
        elif stage == "tune-reg":
            run_tuning("reg", args.models, args.trials)
        elif stage == "tune-clf":
            run_tuning("clf", args.models, args.trials)
        elif stage == "eval-reg":
            run_regression_evaluation()
        elif stage == "eval-clf":
            run_classification_evaluation(explain=not args.no_explain)


if __name__ == "__main__":
    main()
