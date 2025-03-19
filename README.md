# CUDA Kernels

A Python package containing CUDA-accelerated functions for autocorrelation and sum reduction operations.

## Installation

### Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (version 11.0 or higher recommended)
- Python 3.6+
- NumPy

### Installing from PyPI
```bash
pip install cuda-kernels
```

### Installing from source
```bash
git clone https://github.com/AstuteFern/cuda-toolkit.git
cd cuda-toolkit
pip install .
```

## Usage

### Autocorrelation

```python
import numpy as np
from cuda_kernels.autocorrelation import autocorrelation

# Create some test data
data = np.random.randn(10000).astype(np.float32)

# Compute autocorrelation for lags 0 to 100
result = autocorrelation(data, max_lag=100)
```

### Sum Reduction

```python
import numpy as np
from cuda_kernels.reduction import reduction_sum

# Create some test data
data = np.random.randn(10000).astype(np.float32)

# Compute sum
result = reduction_sum(data)
print(f"Sum: {result}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.