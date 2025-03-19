import pytest
import torch
from unittest.mock import patch

from src.utils.gpu_utils import estimate_max_batch_size
from tests.gpu_utils.simple_model import SimpleModel


def test_estimate_max_batch_size_mock():
    """Test batch size estimation with mocked memory stats."""
    model = SimpleModel()
    input_shape = (10,)
    device = torch.device("cuda:0")
    
    # Mock memory stats to simulate 8GB free, 1GB per sample
    with patch('torch.cuda.max_memory_allocated', return_value=4 * 1024**3):  # 4 GB for 4 samples
        with patch('src.utils.gpu_utils.get_gpu_memory_info', return_value={
            0: {'free': 8.0}  # 8 GB free
        }):
            # This should estimate ~8 samples can fit (8GB free / 1GB per sample)
            batch_size = estimate_max_batch_size(model, input_shape, device, max_memory_fraction=1.0)
            assert batch_size >= 7  # Allow some flexibility in the estimate
