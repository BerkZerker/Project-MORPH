import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
from src.utils.distributed import ExpertParallelWrapper
from tests.distributed.simple_model import SimpleModel


def test_expert_parallel_wrapper():
    """Test ExpertParallelWrapper with mocked experts."""
    model = SimpleModel()
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    
    # Mock distribute_experts_across_gpus
    expert_device_map = {0: devices[0], 1: devices[1], 2: devices[0]}
    
    with patch('src.utils.distributed.distribute_experts_across_gpus', return_value=expert_device_map):
        # Mock expert.to() method
        with patch.object(nn.Linear, 'to', return_value=nn.Linear(10, 5)):
            wrapper = ExpertParallelWrapper(model, devices)
            
            # Check that expert_device_map was set
            assert wrapper.expert_device_map == expert_device_map
            assert model.config.expert_device_map == expert_device_map
            
            # Test forward pass with mocked methods
            x = torch.randn(4, 10)
            
            # Mock gating network output
            mock_routing = torch.randn(4, 2)  # k=2
            mock_indices = torch.tensor([[0, 1], [1, 2], [0, 2], [1, 0]])
            mock_uncertainty = torch.tensor(0.1)
            
            with patch.object(model.gating, '__call__', return_value=(mock_routing, mock_indices, mock_uncertainty)):
                # Mock expert outputs
                with patch.object(nn.Linear, '__call__', return_value=torch.randn(4, 5)):
                    # Mock tensor.to() method to avoid actual device transfers
                    with patch.object(torch.Tensor, 'to', return_value=torch.randn(4, 5)):
                        output = wrapper(x)
                        
                        # Check output shape
                        assert output.shape == (4, 5)
