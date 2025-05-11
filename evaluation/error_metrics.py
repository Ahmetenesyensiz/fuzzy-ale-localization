# evaluation/error_metrics.py

import numpy as np

def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error (MAE) hesapla.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE) hesapla.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
