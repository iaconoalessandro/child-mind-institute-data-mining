# ============================================================================================
# PULSAR: Advancing Interval-Based Time Series Classification to State-of-the-Art Performance
# Authors: Nestor Cabello, Lars Kulik
# Reference: IEEE International Conference on Data Mining (ICDM), 2025
# ============================================================================================


from numba import jit, prange
import numpy as np


@jit(
    "Tuple((float32[:], float32[:]))(float32[:,:], int32)",
    nopython=True, fastmath=True, cache=True)
def approx_median_iqr(X, bins):
    """
    Approximate the median and IQR for each row of X using a histogram-based method.
    
    Parameters
    ----------
    X : np.ndarray
        2D array with shape (nrows, ncols)
    bins : int
        Number of bins to use for the histogram.
    
    Returns
    -------
    medians : np.ndarray
        Approximated medians, shape (nrows,).
    iqrs : np.ndarray
        Approximated IQRs, shape (nrows,).
    """

    nrows, ncols = X.shape
    medians = np.empty(nrows, dtype=np.float32)
    iqrs = np.empty(nrows, dtype=np.float32)
    
    for i in prange(nrows):
        row = X[i, :]
        # Determine min and max for the row
        row_min = row[0]
        row_max = row[0]
        for j in range(ncols):
            val = row[j]
            if val < row_min:
                row_min = val
            if val > row_max:
                row_max = val
        
        # If all values are the same, quantiles are trivial.
        if row_max == row_min:
            medians[i] = row_min
            iqrs[i] = 0.0
            continue
        
        # Compute bin width
        width = (row_max - row_min) / bins
        
        # Build histogram (bins are 0-indexed)
        hist = np.zeros(bins, dtype=np.int32)
        for j in range(ncols):
            # Compute bin index; ensure the max value falls into the last bin.
            idx = int((row[j] - row_min) / width)
            if idx >= bins:
                idx = bins - 1
            hist[idx] += 1
        
        # Compute cumulative histogram and find bin indices for 25th, 50th, 75th percentiles.
        cum = 0
        q1_bin = -1
        med_bin = -1
        q3_bin = -1
        for b in range(bins):
            cum += hist[b]
            if q1_bin < 0 and cum >= ncols * 0.25:
                q1_bin = b
            if med_bin < 0 and cum >= ncols * 0.50:
                med_bin = b
            if q3_bin < 0 and cum >= ncols * 0.75:
                q3_bin = b
        
        # Use the midpoint of the bin as the approximate quantile value.
        med_val = row_min + (med_bin + 0.5) * width
        q1_val = row_min + (q1_bin + 0.5) * width
        q3_val = row_min + (q3_bin + 0.5) * width
        
        medians[i] = med_val
        iqrs[i] = q3_val - q1_val
        
    return medians, iqrs



@jit(
    "float32[:,:](float32[:,:])",
    nopython=True, fastmath=True, cache=True)
def partial_local_stats(X):
    """
    Compute these 5 stats in one pass over each row:
      0: mean
      1: stdev
      2: slope
      3: min
      4: max
    X shape: (nrows, ncols)
    Returns: (nrows, 5) array.
    
    Optimizations:
      - Precompute sum_x and sum_xx outside the per-row loop.
      - Combine the loop that computes s, s_sq, cur_min, cur_max with that computing sum_xy.
    """

    nrows, ncols = X.shape
    out = np.zeros((nrows, 5), dtype=np.float32)
    
    # Precompute constant sums (only depend on ncols)
    sum_x = 0.0
    sum_xx = 0.0
    for c in range(ncols):
        sum_x += c
        sum_xx += c * c

    for i in prange(nrows):
        row_data = X[i, :]
        s = 0.0
        s_sq = 0.0
        sum_xy = 0.0  # Merged into the same loop.
        cur_min = row_data[0]
        cur_max = row_data[0]
        for j in range(ncols):
            val = row_data[j]
            s += val
            s_sq += val * val
            sum_xy += j * val
            if val < cur_min:
                cur_min = val
            if val > cur_max:
                cur_max = val

        mean_val = s / ncols
        variance = (s_sq / ncols) - (mean_val * mean_val)
        stdev_val = 0.0
        if variance > 1e-14:
            stdev_val = np.sqrt(variance)
            
        numerator = sum_x * s - ncols * sum_xy
        denominator = (sum_x * sum_x) - (ncols * sum_xx)
        slope_val = 0.0
        if abs(denominator) > 1e-14:
            slope_val = numerator / denominator

        out[i, 0] = mean_val
        out[i, 1] = stdev_val
        out[i, 2] = slope_val
        out[i, 3] = cur_min
        out[i, 4] = cur_max

    return out



def combined_7_local_stats(X):
    """
    Returns shape (nrows, 7) in this order:
       0: mean
       1: stdev
       2: slope
       3: min
       4: max
       5: median
       6: iqr
    """
    # Ensure X is float32
    X = X.astype(np.float32)

    # single pass for 5 stats
    partial_5 = partial_local_stats(X)  # shape (nrows, 5)

    # approximate median and IQR for faster computation
    median_vals, iqr_vals = approx_median_iqr(X, 64)

    # Combine results
    # final: (nrows, 7)
    out = np.column_stack([partial_5, median_vals, iqr_vals])
    return out.astype(np.float32)


class AggregationFunctions:
    def __init__(self):
        pass


    @staticmethod
    def transform_multiple(X, local_stats):
        """
        Compute the requested aggregation statistics for X.
        
        Parameters
        ----------
        X : np.ndarray
            2D array where each row is a sample.
        feature_list : list of str
            The list of feature names to compute.
        
        Returns
        -------
        np.ndarray
            Array containing only the selected features.
        """

        FEATURE_INDEX_MAPPING = {
            "mean": 0,
            "stdev": 1,
            "slope": 2,
            "min": 3,
            "max": 4,
            "median": 5,
            "iqr": 6
        }
        
        # Compute all 7 features
        all_features = combined_7_local_stats(X)

        # Convert feature names to indices
        selected_indices = [FEATURE_INDEX_MAPPING[f] for f in local_stats if f in FEATURE_INDEX_MAPPING]

        # Select only requested columns
        return all_features[:, selected_indices]


