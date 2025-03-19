import os
import numpy as np
import ctypes
from pathlib import Path

_curr_dir = Path(__file__).parent
_lib_path = _curr_dir / "../.." / "_cuda_kernels_autocorrelation_cuda.so"

if not _lib_path.exists():
    raise ImportError(f"Cannot find CUDA kernel at {_lib_path}")

_cuda_lib = ctypes.CDLL(str(_lib_path))

_cuda_lib.run_autocorrelation.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int
]
_cuda_lib.run_autocorrelation.restype = None

def autocorrelation(data, max_lag=None):
    """
    Compute autocorrelation of a time series using CUDA acceleration.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input time series data (must be float32)
    max_lag : int, optional
        Maximum lag to compute. If None, set to min(len(data), 100)
    
    Returns:
    --------
    numpy.ndarray
        Autocorrelation values for lags [0, max_lag)
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    size = len(data)
    
    if max_lag is None:
        max_lag = min(size, 100)
    
    result = np.zeros(max_lag, dtype=np.float32)
    
    # Convert to ctypes pointers
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call the CUDA function
    _cuda_lib.run_autocorrelation(data_ptr, result_ptr, size, max_lag)
    
    return result