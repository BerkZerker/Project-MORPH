import pytest
import torch
from unittest.mock import patch
from src.utils.distributed import setup_distributed_environment


def test_setup_distributed_environment():
    """Test setup_distributed_environment function."""
    # Test with world_size=1 (no distributed)
    with patch('torch.distributed.init_process_group') as mock_init:
        setup_distributed_environment(rank=0, world_size=1)
        mock_init.assert_not_called()  # Should not initialize process group
    
    # Test with world_size>1 (distributed)
    with patch('torch.distributed.init_process_group') as mock_init:
        with patch('torch.cuda.set_device') as mock_set_device:
            setup_distributed_environment(rank=1, world_size=4)
            
            # Should initialize process group
            mock_init.assert_called_once()
            
            # Should set device to rank
            mock_set_device.assert_called_once_with(1)
