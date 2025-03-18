import pytest
import torch
from unittest.mock import patch

from src.utils.gpu_utils import set_gpu_memory_fraction


def test_set_gpu_memory_fraction():
    """Test setting GPU memory fraction."""
    # This is mostly a coverage test since the function may not do anything
    # depending on PyTorch version
    with patch('torch.cuda.is_available', return_value=False):
        # Should not raise an error when CUDA is not available
        set_gpu_memory_fraction(0.5)
    
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.device_count', return_value=2):
            with patch('torch.cuda.set_per_process_memory_fraction') as mock_set:
                set_gpu_memory_fraction(0.5)
                # Should be called once per GPU
                assert mock_set.call_count == 2
