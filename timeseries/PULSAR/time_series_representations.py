# ============================================================================================
# PULSAR: Advancing Interval-Based Time Series Classification to State-of-the-Art Performance
# Authors: Nestor Cabello, Lars Kulik
# Reference: IEEE International Conference on Data Mining (ICDM), 2025
# ============================================================================================


import numpy as np
import pyfftw
from statsmodels.regression.linear_model import burg


class TimeSeriesRepresentations:
    def __init__(self):
        pass

    @staticmethod
    def periodogram_representation(X):
        nfeats = X.shape[1]
        fft_object = pyfftw.builders.fft(X)
        per_X = np.abs(fft_object())
        return per_X[:, :int(nfeats / 2)]

    @staticmethod
    def derivative_representation(X):
        return np.diff(X, axis=1)  
    
    

    @staticmethod
    def autoregressive_representation(X):
        X_transform = []
        lags = int(12*(X.shape[1]/100.)**(1/4.))
        for i in range(X.shape[0]):
            coefs,_ = burg(X[i,:],order=lags)
            X_transform.append(coefs)
        X_transform = np.array(X_transform)
        X_transform[np.isnan(X_transform)] = 0 # In case of Nan values set them to zero    
        return X_transform
    
    
    @staticmethod
    def transform(X, representation):
        representation_map = {
            "original": lambda X:X,
            "periodogram": TimeSeriesRepresentations.periodogram_representation,
            "derivative": TimeSeriesRepresentations.derivative_representation,
            "autoregressive": TimeSeriesRepresentations.autoregressive_representation,
        }

        return representation_map[representation](X)