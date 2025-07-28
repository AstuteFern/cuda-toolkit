import numpy as np
import warnings

def _cpu_reduction_sum(data):
    """CPU fallback implementation of sum reduction."""
    return float(np.sum(data))

# Try to import CUDA implementation, fall back to CPU if not available
try:
    from ._reduction_cuda import run_reduction
    _cuda_available = True
except ImportError:
    _cuda_available = False
    warnings.warn("CUDA extension not available, falling back to CPU implementation", UserWarning)

def reduction_sum(data, force_cpu=False):
    """
    Compute sum reduction using CUDA acceleration when available.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data array (1D float32 array)
    force_cpu : bool, optional
        Force CPU implementation even if CUDA is available
        
    Returns
    -------
    float
        Sum of all elements in the array
        
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
    
    # Use CUDA if available and not forced to use CPU
    if _cuda_available and not force_cpu:
        try:
            result = np.zeros(1, dtype=np.float32)
            run_reduction(data, result, len(data))
            return result[0]
        except Exception as e:
            warnings.warn(f"CUDA implementation failed ({e}), falling back to CPU", UserWarning)
    
    # Fall back to CPU implementation
    return _cpu_reduction_sum(data)

__all__ = ['reduction_sum']