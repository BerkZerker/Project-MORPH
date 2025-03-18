import pytest
import torch
from unittest.mock import patch, MagicMock

from src.utils.gpu_utils import get_optimal_worker_count


def test_get_optimal_worker_count():
    """Test worker count determination."""
    # Test with mocked CPU count
    with patch('multiprocessing.cpu_count', return_value=8):
        with patch('torch.cuda.is_available', return_value=False):
            # CPU only mode
            workers = get_optimal_worker_count()
            assert workers == 4  # Default for CPU
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=2):
                # Mock the GPU properties to avoid CUDA errors
                mock_props = MagicMock()
                mock_props.multi_processor_count = 24  # High enough to get 4 workers per GPU
                
                with patch('torch.cuda.get_device_properties', return_value=mock_props):
                    workers = get_optimal_worker_count()
                    assert workers == 7  # Should be CPU count - 1 (8-1=7)
            
            with patch('torch.cuda.device_count', return_value=4):
                # 4 GPUs, but capped by CPU count
                workers = get_optimal_worker_count()
                assert workers == 8  # Capped at CPU count
