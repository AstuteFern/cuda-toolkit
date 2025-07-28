# CUDA Kernels

A Python package providing CUDA-accelerated functions for autocorrelation and sum reduction operations, with automatic CPU fallback when CUDA is not available.

## Installation

### From PyPI (Recommended)
```bash
pip install cuda-kernels
```

### From GitHub
```bash
pip install git+https://github.com/AstuteFern/cuda-toolkit.git
```

### From Source
```bash
git clone https://github.com/AstuteFern/cuda-toolkit.git
cd cuda-toolkit
pip install .
```

## Requirements

- **Python 3.6+**
- **NumPy**

### Optional (for CUDA acceleration)
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 11.0+)

**Note:** The package works on any system. If CUDA is not available, it automatically uses optimized CPU implementations.

## Quick Start

```python
import numpy as np
from cuda_kernels import autocorrelation, reduction_sum

# Create test data
data = np.random.randn(1000).astype(np.float32)

# Compute autocorrelation (automatically uses CUDA if available)
acf = autocorrelation(data, max_lag=50)
print(f"Autocorrelation shape: {acf.shape}")

# Compute sum reduction
total = reduction_sum(data)
print(f"Sum: {total}")
```

## API Reference

### `autocorrelation(data, max_lag=None, force_cpu=False)`

Compute autocorrelation of a time series.

**Parameters:**
- `data` (numpy.ndarray): Input 1D array (converted to float32)
- `max_lag` (int, optional): Maximum lag to compute. Default: len(data)-1
- `force_cpu` (bool): Force CPU implementation. Default: False

**Returns:**
- `numpy.ndarray`: Autocorrelation values for lags [0, max_lag)

### `reduction_sum(data, force_cpu=False)`

Compute sum of array elements.

**Parameters:**
- `data` (numpy.ndarray): Input 1D array (converted to float32)
- `force_cpu` (bool): Force CPU implementation. Default: False

**Returns:**
- `float`: Sum of all elements

## Examples

### Basic Usage
```python
import numpy as np
from cuda_kernels import autocorrelation, reduction_sum

# Example 1: Autocorrelation
signal = np.sin(np.linspace(0, 4*np.pi, 1000)).astype(np.float32)
acf = autocorrelation(signal, max_lag=100)

# Example 2: Sum reduction
data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
total = reduction_sum(data)  # Returns 15.0
```

### Checking CUDA Status
```python
import sys
autocorr_module = sys.modules['cuda_kernels.autocorrelation']
reduction_module = sys.modules['cuda_kernels.reduction']

print(f"CUDA available: {autocorr_module._cuda_available}")
```

### Force CPU Mode
```python
# Useful for testing or when you want consistent behavior
cpu_result = reduction_sum(data, force_cpu=True)
```

## Performance

- **With CUDA**: Significant speedup for large arrays (10K+ elements)
- **CPU Fallback**: Optimized NumPy implementations, still efficient for most use cases
- **Automatic Detection**: No configuration needed, works out of the box

## License

MIT License - see LICENSE file for details.