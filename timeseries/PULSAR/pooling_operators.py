# ============================================================================================
# PULSAR: Advancing Interval-Based Time Series Classification to State-of-the-Art Performance
# Authors: Nestor Cabello, Lars Kulik
# Reference: IEEE International Conference on Data Mining (ICDM), 2025
# ============================================================================================

from numba import jit, prange, njit
import numpy as np



# Compute Mean Pooling (Using Original Summation Approach)
@njit(fastmath=True, cache=True)
def mean_pooling(X):
    nrows, ncols = X.shape
    out = np.zeros(nrows, dtype=np.float32)

    for i in prange(nrows):
        row = X[i, :]
        s = 0.0
        for j in range(ncols):
            s += row[j]
        out[i] = s / ncols

    return out

# Compute Standard Deviation Pooling (Using Original Variance Calculation)
@njit(fastmath=True, cache=True)
def std_pooling(X):
    nrows, ncols = X.shape
    out = np.zeros(nrows, dtype=np.float32)

    for i in prange(nrows):
        row = X[i, :]
        s = 0.0
        s_sq = 0.0
        for j in range(ncols):
            s += row[j]
            s_sq += row[j] * row[j]
        mean_val = s / ncols
        variance = (s_sq / ncols) - (mean_val * mean_val)
        out[i] = np.sqrt(variance) if variance > 1e-6 else 0.0

    return out

# Compute Slope Pooling (Using Original Summation Approach)
@njit(fastmath=True, cache=True)
def slope_pooling(X):
    nrows, ncols = X.shape
    out = np.zeros(nrows, dtype=np.float32)
    
    sum_x = 0.0
    sum_xx = 0.0
    for j in range(ncols):
        sum_x += j
        sum_xx += j * j

    for i in prange(nrows):
        row = X[i, :]
        s = 0.0
        sum_xy = 0.0
        for j in range(ncols):
            s += row[j]
            sum_xy += j * row[j]

        numerator = sum_x * s - ncols * sum_xy
        denominator = (sum_x * sum_x) - ncols * sum_xx
        out[i] = numerator / denominator if abs(denominator) > 1e-6 else 0.0

    return out

# Compute Minimum Pooling
@njit(fastmath=True, cache=True)
def min_pooling(X):
    nrows, ncols = X.shape
    out = np.zeros(nrows, dtype=np.float32)
    for i in prange(nrows):
        row = X[i, :]
        min_val = row[0]
        for j in range(ncols):
            if row[j] < min_val:
                min_val = row[j]
        out[i] = min_val
    return out

# Compute Maximum Pooling
@njit(fastmath=True, cache=True)
def max_pooling(X):
    nrows, ncols = X.shape
    out = np.zeros(nrows, dtype=np.float32)
    for i in prange(nrows):
        row = X[i, :]
        max_val = row[0]
        for j in range(ncols):
            if row[j] > max_val:
                max_val = row[j]
        out[i] = max_val
    return out

# Compute Mean-Crossing Count Pooling (Using Two-Pass Approach)
@njit(fastmath=True, cache=True)
def mean_crossing_pooling(X):
    nrows, ncols = X.shape
    out = np.zeros(nrows, dtype=np.float32)

    for i in prange(nrows):
        row = X[i, :]
        s = 0.0
        for j in range(ncols):
            s += row[j]
        mean_val = s / ncols

        crossings = 0
        last_val = row[0]
        for j in range(ncols):
            if (last_val <= mean_val and row[j] > mean_val) or (last_val >= mean_val and row[j] < mean_val):
                crossings += 1
            last_val = row[j]

        out[i] = crossings / (ncols - 1) if ncols > 1 else 0.0

    return out

# Compute Values-Above-Mean Pooling (Using Two-Pass Approach)
@njit(fastmath=True, cache=True)
def above_mean_pooling(X):
    nrows, ncols = X.shape
    out = np.zeros(nrows, dtype=np.float32)

    for i in prange(nrows):
        row = X[i, :]
        s = 0.0
        for j in range(ncols):
            s += row[j]
        mean_val = s / ncols

        above_count = 0
        for j in range(ncols):
            if row[j] > mean_val:
                above_count += 1

        out[i] = above_count / ncols if ncols > 0 else 0.0

    return out



# Compute Approximate Median using Histogram-based Method
@njit(fastmath=True, cache=True)
def approx_median(X, bins=64):
    """
    Approximate the median for each row of X using a histogram-based method.
    X is a 2D array of shape (nrows, ncols) with float32 values.
    
    Returns an array of shape (nrows,) with float32 values representing the median.
    """
    nrows, ncols = X.shape
    medians = np.empty(nrows, dtype=np.float32)
    
    for i in range(nrows):
        row = X[i, :]
        row_min = row[0]
        row_max = row[0]
        
        # Find min and max values
        for j in range(ncols):
            val = row[j]
            if val < row_min:
                row_min = val
            if val > row_max:
                row_max = val
        
        if row_max == row_min:
            medians[i] = row_min
            continue
        
        # Compute histogram bins
        width = (row_max - row_min) / bins
        hist = np.zeros(bins, dtype=np.int32)
        for j in range(ncols):
            idx = int((row[j] - row_min) / width)
            if idx >= bins:
                idx = bins - 1
            hist[idx] += 1
        
        # Compute cumulative histogram for median
        cum = 0
        med_bin = -1
        for b in range(bins):
            cum += hist[b]
            if med_bin < 0 and cum >= ncols * 0.50:
                med_bin = b
                break
        
        medians[i] = row_min + (med_bin + 0.5) * width  # Approximate median
    
    return medians


# Compute Approximate Interquartile Range (IQR) using Histogram-based Method
@njit(fastmath=True, cache=True)
def approx_iqr(X, bins=64):
    """
    Approximate the IQR (Interquartile Range) for each row of X using a histogram-based method.
    X is a 2D array of shape (nrows, ncols) with float32 values.
    
    Returns an array of shape (nrows,) with float32 values representing the IQR.
    """
    nrows, ncols = X.shape
    iqrs = np.empty(nrows, dtype=np.float32)
    
    for i in range(nrows):
        row = X[i, :]
        row_min = row[0]
        row_max = row[0]
        
        # Find min and max values
        for j in range(ncols):
            val = row[j]
            if val < row_min:
                row_min = val
            if val > row_max:
                row_max = val
        
        if row_max == row_min:
            iqrs[i] = 0.0
            continue
        
        # Compute histogram bins
        width = (row_max - row_min) / bins
        hist = np.zeros(bins, dtype=np.int32)
        for j in range(ncols):
            idx = int((row[j] - row_min) / width)
            if idx >= bins:
                idx = bins - 1
            hist[idx] += 1
        
        # Compute cumulative histogram for Q1 and Q3
        cum = 0
        q1_bin = -1
        q3_bin = -1
        for b in range(bins):
            cum += hist[b]
            if q1_bin < 0 and cum >= ncols * 0.25:
                q1_bin = b
            if q3_bin < 0 and cum >= ncols * 0.75:
                q3_bin = b
            if q1_bin >= 0 and q3_bin >= 0:
                break

        q1_val = row_min + (q1_bin + 0.5) * width  # Approximate Q1
        q3_val = row_min + (q3_bin + 0.5) * width  # Approximate Q3

        iqrs[i] = q3_val - q1_val  # Compute IQR
    
    return iqrs






class PoolingOperators:
    def __init__(self):
        pass

    @staticmethod
    def transform_individual(X, selected_operator):
        """
        Apply the combined pooling approach on X and return the pooled features
        for the given list of operators.
        
        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (num_examples, segment_length),
            where num_stats is the number of local features per segment.
        selected_operator : list of str
            A pooling operator name. For example, "mean_pooling", "max_pooling", ....
        
        Returns
        -------
        pooled_result : np.ndarray
            A 2D array of shape (num_examples, num_stats * len(selected_operators)).
            For each operator, a 2D array of shape (num_examples, num_stats) is obtained,
            and the results are concatenated horizontally.
        """


        pooling_operators = {
            "mean_pooling": mean_pooling,
            "stdev_pooling": std_pooling,
            "slope_pooling": slope_pooling,
            "min_pooling": min_pooling,
            "max_pooling": max_pooling,
            "mean_crossing_pooling": mean_crossing_pooling,
            "values_above_mean_pooling": above_mean_pooling,
            "median_pooling": approx_median,
            "iqr_pooling": approx_iqr,
        }

        return pooling_operators[selected_operator](X)

