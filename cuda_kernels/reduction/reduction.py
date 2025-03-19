import os
import numpy as np
import ctypes
from pathlib import Path

_curr_dir = Path(__file__).parent
_lib_path = _curr_dir / "../.." / "_cuda_kernels_reduction_cuda.so"

if not _lib_path.exists():
    raise ImportError(f"Cannot find CUDA kernel at {_lib_path}")

_cuda_lib = ctypes.CDLL(str(_lib_path))

_cuda_lib.gpu_reduction_sum.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float)
]
_cuda_lib.gpu_reduction_sum.restype = None

def reduction_sum(data):
    """
    Compute sum of an array using CUDA acceleration.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input array (must be float32)
    
    Returns:
    --------
    float
        Sum of all elements in the array
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    size = len(data)
    result = np.zeros(1, dtype=np.float32)
    
    # Convert to ctypes pointers
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call the CUDA function
    _cuda_lib.gpu_reduction_sum(data_ptr, size, result_ptr)
    
    return result[0]