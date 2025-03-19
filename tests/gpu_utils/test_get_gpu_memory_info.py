import pytest
import torch
from unittest.mock import patch, MagicMock

from src.utils.gpu_utils import get_gpu_memory_info


def test_get_gpu_memory_info():
    """Test GPU memory info retrieval with mocked GPU properties."""
    # Test when CUDA is not available
    with patch('torch.cuda.is_available', return_value=False):
        memory_info = get_gpu_memory_info()
        assert memory_info == {}
    
    # Test with mocked GPU properties
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.device_count', return_value=1):
            # Mock GPU properties
            mock_props = MagicMock()
            mock_props.total_memory = 8 * 1024**3  # 8 GB
            
            with patch('torch.cuda.get_device_properties', return_value=mock_props):
                with patch('torch.cuda.memory_allocated', return_value=1 * 1024**3):  # 1 GB
                    with patch('torch.cuda.memory_reserved', return_value=1 * 1024**3):  # 1 GB
                        memory_info = get_gpu_memory_info()
                        
                        assert 0 in memory_info
                        assert abs(memory_info[0]['total'] - 8.0) < 0.1  # ~8 GB total
                        assert abs(memory_info[0]['free'] - 6.0) < 0.1   # ~6 GB free
                        assert abs(memory_info[0]['used'] - 2.0) < 0.1   # ~2 GB used
