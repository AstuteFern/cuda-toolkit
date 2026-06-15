import ctypes
import os
import warnings

import numpy as np

def _cpu_reduction_sum(data):
    """CPU fallback implementation of sum reduction."""
    return float(np.sum(data))

# Load the compiled CUDA kernel via ctypes, falling back to CPU if the shared
# library was not built (CPU-only install) or cannot be loaded.
_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
try:
    from .._cuda_loader import load_library
    _lib = load_library(os.path.dirname(__file__), "_reduction_cuda")
    # int run_reduction(const float* data, int size, float* result)
    _lib.run_reduction.argtypes = [_FLOAT_PTR, ctypes.c_int, _FLOAT_PTR]
    _lib.run_reduction.restype = ctypes.c_int
    _cuda_available = True
except OSError:
    _lib = None
    _cuda_available = False

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
            data = np.ascontiguousarray(data, dtype=np.float32)
            result = np.zeros(1, dtype=np.float32)
            status = _lib.run_reduction(
                data.ctypes.data_as(_FLOAT_PTR),
                ctypes.c_int(len(data)),
                result.ctypes.data_as(_FLOAT_PTR),
            )
            if status != 0:
                raise RuntimeError(f"run_reduction returned status {status}")
            return float(result[0])
        except Exception as e:
            warnings.warn(f"CUDA implementation failed ({e}), falling back to CPU", UserWarning)
    
    # Fall back to CPU implementation
    return _cpu_reduction_sum(data)

__all__ = ['reduction_sum']