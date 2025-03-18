import pytest
import torch
from unittest.mock import patch

from src.utils.gpu_utils import setup_gpu_environment


def test_setup_gpu_environment():
    """Test GPU environment setup."""
    # This is mostly a coverage test
    with patch('torch.cuda.is_available', return_value=False):
        # Should not raise an error when CUDA is not available
        setup_gpu_environment()
    
    # Test with CUDA available but avoid patching properties directly
    with patch('torch.cuda.is_available', return_value=True):
        with patch('src.utils.gpu_utils.set_gpu_memory_fraction') as mock_set:
            with patch('src.utils.gpu_utils.clear_gpu_cache') as mock_clear:
                # Mock torch.cuda.device_count to return 0 to avoid the loop that accesses memory_clock_rate
                with patch('torch.cuda.device_count', return_value=0):
                    setup_gpu_environment()
                    # Should call set_gpu_memory_fraction
                    mock_set.assert_called_once()
                    # Should call clear_gpu_cache
                    mock_clear.assert_called_once()
