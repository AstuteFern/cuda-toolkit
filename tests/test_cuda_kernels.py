import numpy as np
import pytest
from cuda_kernels import autocorrelation, sum_reduction

def test_autocorrelation_basic():
    """Test basic autocorrelation functionality"""
    # Create test data
    data = np.random.rand(1000).astype(np.float32)
    
    # Compute autocorrelation
    result = autocorrelation(data, max_lag=10)
    
    # Basic checks
    assert result.shape == (10,)
    assert result.dtype == np.float32
    assert not np.any(np.isnan(result))
    
    # Check that lag 0 is the sum of squares
    assert np.isclose(result[0], np.sum(data * data), rtol=1e-5)

def test_autocorrelation_edge_cases():
    """Test autocorrelation with edge cases"""
    # Test with small array
    small_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = autocorrelation(small_data, max_lag=2)
    assert result.shape == (2,)
    
    # Test with all zeros
    zero_data = np.zeros(100, dtype=np.float32)
    result = autocorrelation(zero_data, max_lag=5)
    assert np.all(result == 0)
    
    # Test with all ones
    ones_data = np.ones(100, dtype=np.float32)
    result = autocorrelation(ones_data, max_lag=5)
    assert np.all(result == 100)

def test_autocorrelation_input_validation():
    """Test input validation for autocorrelation"""
    # Test non-1D input
    with pytest.raises(ValueError):
        data_2d = np.random.rand(10, 10).astype(np.float32)
        autocorrelation(data_2d)
    
    # Test non-float32 input
    data_int = np.random.randint(0, 100, 1000)
    result = autocorrelation(data_int)  # Should work as it converts to float32
    assert result.dtype == np.float32

def test_sum_reduction_basic():
    """Test basic sum reduction functionality"""
    # Create test data
    data = np.random.rand(1000).astype(np.float32)
    
    # Compute sum
    result = sum_reduction(data)
    
    # Compare with numpy sum
    np_result = np.sum(data)
    assert np.isclose(result, np_result, rtol=1e-5)

def test_sum_reduction_edge_cases():
    """Test sum reduction with edge cases"""
    # Test with small array
    small_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = sum_reduction(small_data)
    assert np.isclose(result, 6.0, rtol=1e-5)
    
    # Test with all zeros
    zero_data = np.zeros(100, dtype=np.float32)
    result = sum_reduction(zero_data)
    assert result == 0
    
    # Test with all ones
    ones_data = np.ones(100, dtype=np.float32)
    result = sum_reduction(ones_data)
    assert result == 100

def test_sum_reduction_input_validation():
    """Test input validation for sum reduction"""
    # Test non-1D input
    with pytest.raises(ValueError):
        data_2d = np.random.rand(10, 10).astype(np.float32)
        sum_reduction(data_2d)
    
    # Test non-float32 input
    data_int = np.random.randint(0, 100, 1000)
    result = sum_reduction(data_int)  # Should work as it converts to float32
    assert isinstance(result, float)

def test_large_arrays():
    """Test both functions with large arrays"""
    # Create large test data
    large_data = np.random.rand(1000000).astype(np.float32)
    
    # Test autocorrelation
    acf_result = autocorrelation(large_data, max_lag=100)
    assert acf_result.shape == (100,)
    assert not np.any(np.isnan(acf_result))
    
    # Test sum reduction
    sum_result = sum_reduction(large_data)
    np_sum = np.sum(large_data)
    assert np.isclose(sum_result, np_sum, rtol=1e-5) 