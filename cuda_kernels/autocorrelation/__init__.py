import numpy as np
from .autocorrelation_cuda import run_autocorrelation

def autocorrelation(data, max_lag=None):
    """
    Compute autocorrelation using CUDA acceleration.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data array (1D float32 array)
    max_lag : int, optional
        Maximum lag to compute. If None, uses len(data) - 1
        
    Returns
    -------
    numpy.ndarray
        Array of autocorrelation values for lags 0 to max_lag
        
    Raises
    ------
    ValueError
        If input data is not 1D or not float32
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
        
    if max_lag is None:
        max_lag = len(data) - 1
        
    result = np.zeros(max_lag, dtype=np.float32)
    run_autocorrelation(data, result, len(data), max_lag)
    return result

__all__ = ['autocorrelation']