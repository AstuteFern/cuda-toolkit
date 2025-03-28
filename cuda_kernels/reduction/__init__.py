import numpy as np
from .reduction_cuda import run_reduction

def sum_reduction(data):
    """
    Compute sum reduction using CUDA acceleration.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data array (1D float32 array)
        
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
        
    result = np.zeros(1, dtype=np.float32)
    run_reduction(data, result, len(data))
    return result[0]

__all__ = ['sum_reduction']