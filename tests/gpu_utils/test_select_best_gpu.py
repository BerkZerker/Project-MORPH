import pytest
import torch
from unittest.mock import patch

from src.utils.gpu_utils import select_best_gpu


def test_select_best_gpu():
    """Test best GPU selection with mocked memory info."""
    # Test when CUDA is not available
    with patch('torch.cuda.is_available', return_value=False):
        best_gpu = select_best_gpu()
        assert best_gpu is None
    
    # Test with mocked memory info
    with patch('torch.cuda.is_available', return_value=True):
        with patch('src.utils.gpu_utils.get_gpu_memory_info', return_value={
            0: {'free': 4.0},
            1: {'free': 6.0},  # This one has more free memory
            2: {'free': 2.0}
        }):
            best_gpu = select_best_gpu()
            assert best_gpu == 1  # Should select GPU 1 with most free memory
