import pytest
import torch
from unittest.mock import patch, MagicMock
from src.utils.distributed import DataParallelWrapper
from tests.distributed.simple_model import SimpleModel


def test_data_parallel_wrapper_mock():
    """Test DataParallelWrapper with mocked DataParallel."""
    model = SimpleModel()
    devices = [torch.device("cuda:0")]
    
    # Mock nn.DataParallel
    mock_dp = MagicMock()
    mock_dp.return_value = torch.randn(4, 5)
    
    with patch('torch.nn.DataParallel', return_value=mock_dp):
        wrapper = DataParallelWrapper(model, devices)
        
        # Test forward pass
        x = torch.randn(4, 10)
        output = wrapper(x)
        
        # Check that DataParallel was called
        mock_dp.assert_called_once()
        
        # Test train_step and evaluate pass-through
        optimizer = MagicMock()
        criterion = MagicMock()
        data_loader = MagicMock()
        
        wrapper.train_step((x, torch.randn(4, 5)), optimizer, criterion)
        wrapper.evaluate(data_loader, criterion)
