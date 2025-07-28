import numpy as np
import pytest
from cuda_kernels import autocorrelation, reduction_sum

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
    # For ones data, autocorr[lag] = (100-lag) since we have (100-lag) overlapping 1*1 products
    expected = np.array([100, 99, 98, 97, 96], dtype=np.float32)
    assert np.allclose(result, expected)

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
    result = reduction_sum(data)
    
    # Compare with numpy sum
    np_result = np.sum(data)
    assert np.isclose(result, np_result, rtol=1e-5)

def test_sum_reduction_edge_cases():
    """Test sum reduction with edge cases"""
    # Test with small array
    small_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = reduction_sum(small_data)
    assert np.isclose(result, 6.0, rtol=1e-5)
    
    # Test with all zeros
    zero_data = np.zeros(100, dtype=np.float32)
    result = reduction_sum(zero_data)
    assert result == 0
    
    # Test with all ones
    ones_data = np.ones(100, dtype=np.float32)
    result = reduction_sum(ones_data)
    assert result == 100

def test_sum_reduction_input_validation():
    """Test input validation for sum reduction"""
    # Test non-1D input
    with pytest.raises(ValueError):
        data_2d = np.random.rand(10, 10).astype(np.float32)
        reduction_sum(data_2d)
    
    # Test non-float32 input
    data_int = np.random.randint(0, 100, 1000)
    result = reduction_sum(data_int)  # Should work as it converts to float32
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
    sum_result = reduction_sum(large_data)
    np_sum = np.sum(large_data)
    assert np.isclose(sum_result, np_sum, rtol=1e-5) 