import numpy as np
import warnings

def _cpu_autocorrelation(data, max_lag):
    """CPU fallback implementation of autocorrelation."""
    result = np.zeros(max_lag, dtype=np.float32)
    for lag in range(max_lag):
        if lag < len(data):
            # Calculate autocorrelation for this lag
            n_valid = len(data) - lag
            if n_valid > 0:
                result[lag] = np.sum(data[:n_valid] * data[lag:lag+n_valid])
            else:
                result[lag] = 0.0
        else:
            result[lag] = 0.0
    return result

# Try to import CUDA implementation, fall back to CPU if not available
try:
    from ._autocorrelation_cuda import run_autocorrelation
    _cuda_available = True
except ImportError:
    _cuda_available = False
    warnings.warn("CUDA extension not available, falling back to CPU implementation", UserWarning)

def autocorrelation(data, max_lag=None, force_cpu=False):
    """
    Compute autocorrelation using CUDA acceleration when available.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data array (1D float32 array)
    max_lag : int, optional
        Maximum lag to compute. If None, uses len(data) - 1
    force_cpu : bool, optional
        Force CPU implementation even if CUDA is available
        
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
        
    # Use CUDA if available and not forced to use CPU
    if _cuda_available and not force_cpu:
        try:
            result = np.zeros(max_lag, dtype=np.float32)
            run_autocorrelation(data, result, len(data), max_lag)
            return result
        except Exception as e:
            warnings.warn(f"CUDA implementation failed ({e}), falling back to CPU", UserWarning)
    
    # Fall back to CPU implementation
    return _cpu_autocorrelation(data, max_lag)

__all__ = ['autocorrelation']