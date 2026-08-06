# ============================================================================================
# PULSAR: Advancing Interval-Based Time Series Classification to State-of-the-Art Performance
# Authors: Nestor Cabello, Lars Kulik
# Reference: IEEE International Conference on Data Mining (ICDM), 2025
# ============================================================================================


from numba import jit
import numpy as np


@jit(nopython=True, fastmath=True)
def _calculate_mean(X):
    ncols = len(X)
    accum = 0
    for i in range(ncols):
        accum += X[i]
    return accum / ncols

## This version of stdev uses the already computed mean
@jit(nopython=True, fastmath=True)
def _calculate_stdev(X, Xmean): #Sample Standard Deviation (similar to np.std(ddof=1))
    ncols = len(X)
    accum = 0
    for i in range(ncols):
        accum+=(X[i]-Xmean)**2
    if ncols==1:
        return 0.00001
    else:
        return (accum/(ncols-1))**0.5

class FeatureSelection:
    def __init__(self):
        pass    

    @staticmethod
    @jit(nopython=True, fastmath = True)
    def fisher_score(X_feat, y):
        unique_labels = np.unique(y)
        mu_feat = _calculate_mean(X_feat)
        accum_numerator = 0
        accum_denominator = 0
        
        for k in unique_labels:
            idx_label = np.where(y==k)[0]
            nk = len(idx_label)
            data_sub = X_feat[idx_label]

            mu_feat_label = _calculate_mean(data_sub)
            sigma_feat_label = max(_calculate_stdev(data_sub, mu_feat_label),0.0001) ###to avoid div by zero in case 1 class label per instance      
            
            accum_numerator += nk*(mu_feat_label-mu_feat)**2
            accum_denominator +=  nk*sigma_feat_label**2
        if accum_numerator==0 or accum_denominator==0:
            return 0
        else:
            return accum_numerator/accum_denominator
        
    