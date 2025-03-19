import pytest
import torch
from unittest.mock import patch

from src.utils.gpu_utils import detect_and_select_gpu


def test_detect_and_select_gpu():
    """Test GPU mode and device selection."""
    # Test when CUDA is not available
    with patch('torch.cuda.is_available', return_value=False):
        mode, devices = detect_and_select_gpu()
        assert mode == "cpu"
        assert len(devices) == 1
        assert devices[0].type == "cpu"
    
    # Test with single GPU
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.device_count', return_value=1):
            mode, devices = detect_and_select_gpu()
            assert mode == "single_gpu"
            assert len(devices) == 1
            assert devices[0].type == "cuda"
            assert devices[0].index == 0
    
    # Test with multiple GPUs
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.device_count', return_value=2):
            mode, devices = detect_and_select_gpu()
            assert mode == "multi_gpu"
            assert len(devices) == 2
            assert all(d.type == "cuda" for d in devices)
            assert devices[0].index == 0
            assert devices[1].index == 1
