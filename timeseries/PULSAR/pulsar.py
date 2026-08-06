# ============================================================================================
# PULSAR: Advancing Interval-Based Time Series Classification to State-of-the-Art Performance
# Authors: Nestor Cabello, Lars Kulik
# Reference: IEEE International Conference on Data Mining (ICDM), 2025
# ============================================================================================


from utils import generate_fixed_intervals, compute_global_and_local_features_train, compute_global_and_local_features_test
from feature_selection import FeatureSelection
from time_series_representations import TimeSeriesRepresentations
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.calibration import CalibratedClassifierCV
from collections import Counter
import numpy as np
import pandas as pd
from time import time



class PULSAR():
    def __init__(self, bake_off_classifiers=None, time_series_representations=None, 
                 local_statistics=None, global_statistics=None,
                 list_interval_lengths=None, depth_local_features=None, 
                 percentage_top_local_features=None,
                 num_random_selected_pooling_operators_per_interval=None, 
                 max_dilation=None):
        """
        Initializes the PULSAR classifier.

        Parameters:
        -----------
        bake_off_classifiers : list of str or None
            List of classifier names to use (e.g., ['ridge', 'extra_trees']).
        time_series_representations : list of str or None
            List of time series representations to apply (e.g., ['original', 'derivative']).
        local_statistics : list of str or None
            List of local statistics to compute on intervals (e.g., ['mean', 'stdev']).
        global_statistics : list of str or None
            List of global pooling operators to apply on local features.
        list_interval_lengths : list of int or None
            List of base interval lengths to consider (e.g., [7, 9, 11]).
        depth_local_features : int or None
            Depth for hierarchical feature generation.
        percentage_top_local_features : int or None
            Percentage of top local features to select using Fisher score.
        num_random_selected_pooling_operators_per_interval : int or None
            Number of global pooling operators to randomly select per interval feature set.
        max_dilation : int or None
            Maximum dilation factor for generating intervals.
        """

        self.bake_off_classifiers = (
            ['ridge', 'extra_trees']
            if bake_off_classifiers is None else bake_off_classifiers
        )

        self.time_series_representations = (
            ['original', 'periodogram', 'derivative', 'autoregressive']
            if time_series_representations is None else time_series_representations
        )

        self.local_statistics = (
            ['mean', 'stdev', 'slope', 'min', 'max', 'iqr', 'median']
            if local_statistics is None else local_statistics
        )

        self.global_statistics = (
            [
                'max_pooling', 'mean_pooling', 'min_pooling', 'median_pooling',
                'iqr_pooling', 'stdev_pooling', 'mean_crossing_pooling',
                'values_above_mean_pooling', 'slope_pooling'
            ]
            if global_statistics is None else global_statistics
        )

        self.list_interval_lengths = (
            [7, 9, 11] if list_interval_lengths is None else list_interval_lengths
        )

        self.depth_local_features = (
            4 if depth_local_features is None else depth_local_features
        )

        self.percentage_top_local_features = (
            40 if percentage_top_local_features is None else percentage_top_local_features
        )

        self.num_random_selected_pooling_operators_per_interval = (
            6 if num_random_selected_pooling_operators_per_interval is None
            else num_random_selected_pooling_operators_per_interval
        )

        self.max_dilation = 16 if max_dilation is None else max_dilation

        # Stores interval lengths and dilations for each representation.
        self.lengths_dilations = {}
        # Stores initial feature index for local pooled features per representation.
        self.initial_local_pooled_features_indices = {}
        # Stores indices of top-K selected local features.
        self.top_k_indices = None
        # StandardScaler instance.
        self.scaler = None
        # Dictionary to store fitted classifiers.
        self.classifiers = {}
        # Stores selected pooling operators for each representation.
        self.selected_operators = {}
        # Stores interval partitions for each representation.
        self.partitions = {}
        # Stores hierarchical levels for each representation.
        self.levels = {}
        # List of metadata for local features.
        self.total_list_of_indices_local = []
        # List of metadata for global features.
        self.total_list_of_indices_global = []
        # Combined list of metadata for global and local features.
        self.total_list_of_indices_combined = []
        # Stores test local features before Fisher score selection (for debugging).
        self.X_test_local_before_fisherScore = None
        # Dictionary mapping feature descriptions to details for consistent feature generation.
        self.relevant_features_dictionary = None
        # Cache for the last computed prediction probabilities.
        self.predict_proba_cache = None

    def get_params(self, deep=True):
        """
        Returns parameters as a dictionary for compatibility with scikit-learn.

        Parameters:
        -----------
        deep : bool, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns:
        --------
        dict
            Parameter names mapped to their values.
        """
        # Returns a subset of initialization parameters.
        # Adding all parameters to be complete for scikit-learn
        return {
            'bake_off_classifiers': self.bake_off_classifiers,
            'time_series_representations': self.time_series_representations,
            'local_statistics': self.local_statistics,
            'global_statistics': self.global_statistics,
            'list_interval_lengths': self.list_interval_lengths,
            'depth_local_features': self.depth_local_features,
            'percentage_top_local_features': self.percentage_top_local_features,
            'num_random_selected_pooling_operators_per_interval': self.num_random_selected_pooling_operators_per_interval,
            'max_dilation': self.max_dilation,
        }

    def set_params(self, **params):
        """
        Sets parameters from a dictionary for compatibility with scikit-learn.

        Parameters:
        -----------
        **params : dict
            Estimator parameters.

        Returns:
        --------
        self
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    


    def initialise_classifiers(self, X, y, cv):
        """
        Initialises and fits base classifiers, then calibrates them.

        Parameters:
        -----------
        X : np.ndarray
            The transformed training data (features).
        y : np.ndarray
            The training target values.
        cv : int or str
            Cross-validation strategy for CalibratedClassifierCV.
        """
        print(f'Initialising classifiers: {self.bake_off_classifiers}')

        if 'extra_trees' in self.bake_off_classifiers:
            extra_trees = ExtraTreesClassifier(n_estimators=50, criterion='entropy', max_features=0.10)
            # If cv suggests few folds, prefit the base estimator for calibration.
            if cv < 5:
                extra_trees_cv = 'prefit'
                extra_trees.fit(X, y)
            else:
                extra_trees_cv = cv
            self.classifiers['extra_trees'] = CalibratedClassifierCV(extra_trees, method='sigmoid', cv=extra_trees_cv)

        if 'ridge' in self.bake_off_classifiers:
            ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            # If cv suggests few folds, prefit the base estimator for calibration.
            if cv < 5:
                ridge_cv = 'prefit'
                ridge.fit(X, y)
            else:
                ridge_cv = cv
            self.classifiers['ridge'] = CalibratedClassifierCV(ridge, method='sigmoid', cv=ridge_cv)



    def fit_transform(self, X, y):
        """
        Performs feature engineering on training data `X` and `y`, and learns transformation parameters.

        This includes:
        1. Applying time series representations.
        2. Generating local features from intervals.
        3. Generating global features by pooling local features.
        4. Selecting top local features using Fisher score.
        5. Combining global and selected local features.
        6. Standardizing the combined features.

        This method also learns and stores transformation parameters (e.g., scaler, top_k_indices)
        that will be used by the `transform` method on new data.

        Parameters:
        -----------
        X : np.ndarray
            Training time series data, shape (n_instances, n_timepoints).
        y : np.ndarray
            Training target values, shape (n_instances,).

        Returns:
        --------
        np.ndarray
            The transformed and standardized training feature matrix.
        """
        # Reset lists for feature metadata for the current fit.
        self.total_list_of_indices_local = []
        self.total_list_of_indices_global = []

        # Reset lists for accumulating feature arrays.
        self.X_train_transformed_list = []
        self.global_features_train_list = []

        # Initialize offset for features from compute_global_and_local_features_train.
        initial_local_pooled_feature_index = 0

        n_instances, _ = X.shape # Original input_length is shadowed in loop.

        # Iterate over each specified time series representation.
        for str_representation in self.time_series_representations:
            # Apply the current time series representation.
            X_transformed = TimeSeriesRepresentations.transform(X, str_representation)
            # Update input_length to the length of the current transformed series.
            current_input_length = X_transformed.shape[1] # Use a different variable name.
            # Generate interval configurations for the current representation.
            lengths, dilations = generate_fixed_intervals(current_input_length, self.list_interval_lengths, self.max_dilation)
            # Store configurations for use in transform method.
            self.lengths_dilations[str_representation] = (lengths, dilations)

            # Store initial feature index offset for this representation.
            self.initial_local_pooled_features_indices[str_representation] = initial_local_pooled_feature_index

            # Compute local, global features and related metadata.
            local_train, global_train, selected_operators, list_of_indices_local, list_of_indices_global, lst_partitions, lst_levels = compute_global_and_local_features_train(X_transformed, lengths, dilations, 
                                                                                                                                                                               self.local_statistics, self.global_statistics, 
                                                                                                                                                                               self.depth_local_features,
                                                                                                                                                                               initial_local_pooled_feature_index,
                                                                                                                                                                               self.num_random_selected_pooling_operators_per_interval,
                                                                                                                                                                               str_representation)
            # Store selected pooling operators.
            self.selected_operators[str_representation] = selected_operators

            # Create metadata for raw time series points of the current representation.
            list_of_indices_ts_representation = []
            for i in range(X_transformed.shape[1]):
                list_of_indices_ts_representation.append([-1, str_representation, "", "", "", i, "", "", "", ""])

            # Accumulate metadata for local features (generated and raw).
            self.total_list_of_indices_local.extend(list_of_indices_local)
            self.total_list_of_indices_local.extend(list_of_indices_ts_representation)
            # Accumulate metadata for global features.
            self.total_list_of_indices_global.extend(list_of_indices_global)

            # Store partitions and levels.
            self.partitions[str_representation] = lst_partitions
            self.levels[str_representation] = lst_levels

            # Accumulate feature arrays.
            if len(local_train) > 0: # Check if local_train has any elements.
                initial_local_pooled_feature_index += local_train.shape[1]
                self.X_train_transformed_list.append(local_train)
            self.X_train_transformed_list.append(X_transformed) # Add raw transformed data.
            if global_train.size > 0:
                self.global_features_train_list.append(global_train)

        # Combine all local feature arrays. If list is empty, result is 1D empty array.
        X_train_local = np.hstack(self.X_train_transformed_list) if self.X_train_transformed_list else np.array([])
        # Combine all global feature arrays. If list is empty, result is 1D empty array.
        X_train_global = np.hstack(self.global_features_train_list) if self.global_features_train_list else np.array([])

        # print(f'X_train_local shape: {X_train_local.shape}')
        # print(f'X_train_global shape: {X_train_global.shape}')

        # Apply Fisher Score if local features were generated.
        if X_train_local.size > 0:
            # Compute Fisher scores for each local feature.
            fisher_scores = np.array([FeatureSelection.fisher_score(X_train_local[:, i], y) for i in range(X_train_local.shape[1])])
            # Calculate number of top features (K) to select.
            K = int((self.percentage_top_local_features / 100) * X_train_local.shape[1])
            # Get indices of top K features.
            self.top_k_indices = np.argsort(fisher_scores)[-K:]
            # Select features based on these indices.
            X_train_local_selected = X_train_local[:, self.top_k_indices]

            # Filter metadata list for local features based on selected indices.
            # Assumes self.total_list_of_indices_local items correspond to X_train_local columns.
            filtered_total_list_of_indices_local = []
            for feature_index, feature_value in enumerate(self.total_list_of_indices_local):
                if feature_index in self.top_k_indices: # Check if column index was selected.
                    filtered_total_list_of_indices_local.append(feature_value)
            self.total_list_of_indices_local = filtered_total_list_of_indices_local
        else:
            # If no local features, selected local features are an empty 1D array.
            X_train_local_selected = np.array([])
            # If no local features, top_k_indices should be empty.
            self.top_k_indices = np.array([], dtype=int)


        # Create dictionary of relevant features from (filtered) local feature metadata.
        if self.total_list_of_indices_local:
            columns = [
                "index", "time_series_representation", "interval_length", "interval_dilation",
                "level", "partition_number", "start_partition_idx", "end_partition_idx",
                "global_pooling_operator", "local_statistic"
            ]
            df = pd.DataFrame(self.total_list_of_indices_local, columns=columns)
            unique_localstat_globalstat_pairs_by_representation_interval = df[["time_series_representation", "interval_length", "interval_dilation", "local_statistic", "global_pooling_operator", "start_partition_idx", "end_partition_idx"]].drop_duplicates()
            self.relevant_features_dictionary = {
                f"{row['time_series_representation']},{row['interval_length']},{row['interval_dilation']},{row['local_statistic']},{row['global_pooling_operator']},{row['start_partition_idx']},{row['end_partition_idx']}": row.to_dict()
                for _, row in unique_localstat_globalstat_pairs_by_representation_interval.iterrows()
            }
        else:
            self.relevant_features_dictionary = {}


        # print(f'X_train_local_selected shape: {X_train_local_selected.shape}')

        # Combine selected local features and global features.
        if X_train_local_selected.size > 0:
            # If global features also exist, stack them; otherwise, use only selected local.
            X_train_combined = np.hstack([X_train_global, X_train_local_selected]) if X_train_global.size else X_train_local_selected
        else:
            # If no selected local features, combined features are just global features.
            X_train_combined = X_train_global

        # Standardize the combined features.
        if X_train_combined.size > 0:
            self.scaler = StandardScaler()
            X_train_combined = self.scaler.fit_transform(X_train_combined)
        else:
            self.scaler = None # No scaler if no features.

        # Store combined list of feature metadata.
        self.total_list_of_indices_combined = self.total_list_of_indices_global + self.total_list_of_indices_local

        return X_train_combined

    def transform(self, X):
        """
        Transforms new time series data `X` using parameters learned during `fit_transform`.

        Parameters:
        -----------
        X : np.ndarray
            New time series data to transform, shape (n_instances, n_timepoints).

        Returns:
        --------
        np.ndarray
            The transformed and standardized feature matrix for the new data.
        """
        # Reset lists for accumulating feature arrays for the current test set.
        self.X_test_transformed_list = []
        self.global_features_test_list = []

        # Iterate over each specified time series representation.
        for str_representation in self.time_series_representations:
            # Apply the current time series representation.
            X_transformed = TimeSeriesRepresentations.transform(X, str_representation)
            # Retrieve stored parameters from fitting (will KeyError if representation not seen in fit).
            lengths, dilations = self.lengths_dilations[str_representation]
            selected_operators_per_interval = self.selected_operators[str_representation]
            lst_partitions = self.partitions[str_representation]
            lst_levels = self.levels[str_representation]

            # Compute local and global features for test data.
            local_test, global_test = compute_global_and_local_features_test(X_transformed, lengths, dilations, 
                                                                             self.local_statistics, 
                                                                             self.global_statistics,
                                                                             selected_operators_per_interval,
                                                                             str_representation,
                                                                             lst_partitions, lst_levels,
                                                                             self.relevant_features_dictionary)
            # Accumulate feature arrays.
            if len(local_test) > 0:
                self.X_test_transformed_list.append(local_test)
            self.X_test_transformed_list.append(X_transformed) # Add raw transformed data.
            if global_test.size > 0:
                self.global_features_test_list.append(global_test)

        # Combine all local feature arrays. If list is empty, result is 1D empty array.
        X_test_local = np.hstack(self.X_test_transformed_list) if self.X_test_transformed_list else np.array([])
        # Combine all global feature arrays. If list is empty, result is 1D empty array.
        X_test_global = np.hstack(self.global_features_test_list) if self.global_features_test_list else np.array([])

        # Store all local features before selection.
        self.X_test_local_before_fisherScore = X_test_local

        # Select local features using indices from training.
        if X_test_local.size > 0:
            # Apply stored top_k_indices (will error if not set or incompatible).
            X_test_local_selected = X_test_local[:, self.top_k_indices]
        else:
            # If no local features, selected local features are an empty 1D array.
            X_test_local_selected = np.array([])

        # Combine selected local features and global features.
        if X_test_local_selected.size > 0:
            # If global features also exist, stack them; otherwise, use only selected local.
            X_test_combined = np.hstack([X_test_global, X_test_local_selected]) if X_test_global.size else X_test_local_selected
        else:
            # If no selected local features, combined features are just global features.
            X_test_combined = X_test_global

        # Standardize the combined features using the fitted scaler.
        if self.scaler and X_test_combined.size > 0:
            X_test_combined = self.scaler.transform(X_test_combined)
 

        return X_test_combined

    def fit(self, X, y):
        """
        Fits the RandomPseudoConvolutionalIntervals model to the training data.

        This involves:
        1. Transforming the training data `X` and `y` using `fit_transform` to generate features
           and learn transformation parameters.
        2. Determining an appropriate cross-validation strategy for classifier calibration.
        3. Initializing and fitting the base classifiers (e.g., ExtraTrees, Ridge) on the
           transformed features.

        Parameters:
        -----------
        X : np.ndarray
            Training time series data, shape (n_instances, n_timepoints).
        y : np.ndarray
            Training target values, shape (n_instances,).
        """
        start_time = time()
        # Transform training data and learn transformation parameters.
        X = self.fit_transform(X, y)
        # print(f'Time taken to transform train data: {time() - start_time:.4f}s') # Uncomment for timing info.

        # Determine cross-validation folds for calibration.
        class_counts = Counter(y)
        # This will error if y is empty or class_counts is empty.
        min_class_size = min(class_counts.values()) if class_counts else 0
        cv = min(5, min_class_size) if min_class_size > 0 else 1 # Ensure cv is at least 1

        # Initialize and fit classifiers on transformed data.
        self.initialise_classifiers(X, y, cv) # Pass transformed data

        # Fit each calibrated classifier.
        for name, clf in self.classifiers.items():
            print(f'Fitting {name}')
            clf.fit(X, y) # Fit on transformed data

        # Clear probability cache after fitting.
        self.predict_proba_cache = None

    def predict(self, X):
        """
        Predicts class labels for new data `X`.

        This method first computes class probabilities using `predict_proba` (which involves
        transforming `X`), and then determines the class with the highest probability.
        The computed probabilities are cached.

        Parameters:
        -----------
        X : np.ndarray
            New time series data to predict, shape (n_instances, n_timepoints).

        Returns:
        --------
        np.ndarray
            Predicted class labels, shape (n_instances,).
        """
        # Get class probabilities (this also transforms X and caches probabilities).
        y_pred_proba = self.predict_proba(X)
        # Determine class labels from probabilities.
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred



    def predict_proba(self, X):
        """
        Predicts class probabilities for new data `X`.

        This involves:
        1. Transforming `X` using the learned transformation parameters.
        2. Getting probability predictions from each fitted base classifier.
        3. Averaging probabilities if multiple classifiers are used.
        The computed probabilities are stored in `self.predict_proba_cache`.

        Parameters:
        -----------
        X : np.ndarray
            New time series data, shape (n_instances, n_timepoints).

        Returns:
        --------
        np.ndarray or None
            Predicted class probabilities, shape (n_instances, n_classes).
        """
        start_time = time()
        # Transform test data using learned parameters.
        X = self.transform(X)
        # print(f'Time taken to transform test data: {time() - start_time:.4f}s') # Uncomment for timing info.

        # Collect probabilities from each classifier.
        all_probas = []
        for name, clf in self.classifiers.items():
            print(f'Predicting probabilities with {name}')
            y_clf_proba = clf.predict_proba(X)
            all_probas.append(y_clf_proba)

        # Average probabilities if multiple classifiers were used.
        if len(all_probas) > 1:
            y_final_proba = np.mean(all_probas, axis=0)
        else: # Only one classifier, or only one set of probabilities collected.
            y_final_proba = all_probas[0]

        # Cache the computed probabilities.
        self.predict_proba_cache = y_final_proba
        return y_final_proba

    def get_predict_proba(self):
        """
        Returns the cached prediction probabilities.

        Returns:
        --------
        np.ndarray
            The cached prediction probabilities.

        Raises:
        -------
        ValueError
            If probabilities have not been computed yet via `predict_proba` or `predict`.
        """
        if self.predict_proba_cache is None:
            raise ValueError("Prediction probabilities have not been computed yet. Call predict() or predict_proba() first.")
        return self.predict_proba_cache