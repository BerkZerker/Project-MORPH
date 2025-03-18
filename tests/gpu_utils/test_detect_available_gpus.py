import pytest
import torch
from unittest.mock import patch

from src.utils.gpu_utils import detect_available_gpus


def test_detect_available_gpus():
    """Test GPU detection with mocked CUDA availability."""
    # Test when CUDA is not available
    with patch('torch.cuda.is_available', return_value=False):
        gpus = detect_available_gpus()
        assert gpus == []
    
    # Test when CUDA is available with 2 GPUs
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.device_count', return_value=2):
            gpus = detect_available_gpus()
            assert gpus == [0, 1]
