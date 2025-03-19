import pytest
import torch
from unittest.mock import patch, MagicMock
from src.utils.distributed import create_parallel_wrapper
from tests.distributed.simple_model import SimpleModel


def test_create_parallel_wrapper():
    """Test create_parallel_wrapper function."""
    model = SimpleModel()
    config = MagicMock()
    
    # Test CPU mode
    config.gpu_mode = "cpu"
    config.devices = [torch.device("cpu")]
    wrapped = create_parallel_wrapper(model, config)
    assert wrapped is model  # Should return the original model
    
    # Test single GPU mode
    config.gpu_mode = "single_gpu"
    config.devices = [torch.device("cuda:0")]
    wrapped = create_parallel_wrapper(model, config)
    assert wrapped is model  # Should return the original model
    
    # Test data parallel mode
    config.gpu_mode = "multi_gpu"
    config.devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    config.parallel_strategy = "data_parallel"
    
    with patch('src.utils.distributed.DataParallelWrapper', return_value="data_parallel_model"):
        wrapped = create_parallel_wrapper(model, config)
        assert wrapped == "data_parallel_model"
    
    # Test expert parallel mode
    config.parallel_strategy = "expert_parallel"
    
    with patch('src.utils.distributed.ExpertParallelWrapper', return_value="expert_parallel_model"):
        wrapped = create_parallel_wrapper(model, config)
        assert wrapped == "expert_parallel_model"
    
    # Test unknown strategy
    config.parallel_strategy = "unknown_strategy"
    wrapped = create_parallel_wrapper(model, config)
    assert wrapped is model  # Should return the original model
