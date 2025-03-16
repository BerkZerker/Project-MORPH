import pytest
import torch
import torch.nn as nn
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path to import morph
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from morph.utils.distributed import (
    DataParallelWrapper,
    ExpertParallelWrapper,
    create_parallel_wrapper,
    setup_distributed_environment
)


# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.experts = nn.ModuleList([nn.Linear(10, 5) for _ in range(3)])
        self.gating = nn.Linear(10, 3)
        self.step_count = 0
        self.sleep_module = MagicMock()
        self.knowledge_graph = MagicMock()
        self.config = MagicMock()
        self.config.output_size = 5
        self.config.expert_k = 2
        self.config.devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        self.device = torch.device("cuda:0")
        self.expert_device_map = {i: torch.device("cuda:0") for i in range(3)}
        self.enable_mixed_precision = False
        self.scaler = None
        
    def forward(self, x, training=True):
        return self.fc(x)
    
    def train_step(self, batch, optimizer, criterion):
        inputs, targets = batch
        outputs = self(inputs)
        loss = criterion(outputs, targets)
        return {'loss': loss.item(), 'accuracy': 95.0, 'num_experts': len(self.experts)}
    
    def evaluate(self, data_loader, criterion, device=None):
        return {'loss': 0.1, 'accuracy': 96.0, 'num_experts': len(self.experts)}
    
    def sleep(self):
        pass


class TestDistributed:
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_data_parallel_wrapper_real(self):
        """Test DataParallelWrapper with real GPU if available."""
        # Only run this test if CUDA is actually available
        model = SimpleModel()
        devices = [torch.device("cuda:0")]
        
        # Create wrapper
        wrapper = DataParallelWrapper(model, devices)
        
        # Test forward pass
        x = torch.randn(4, 10).to(devices[0])
        output = wrapper(x)
        
        # Check output shape
        assert output.shape == (4, 5)
    
    def test_data_parallel_wrapper_mock(self):
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
    
    def test_expert_parallel_wrapper(self):
        """Test ExpertParallelWrapper with mocked experts."""
        model = SimpleModel()
        devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        
        # Mock distribute_experts_across_gpus
        expert_device_map = {0: devices[0], 1: devices[1], 2: devices[0]}
        
        with patch('morph.utils.distributed.distribute_experts_across_gpus', return_value=expert_device_map):
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
    
    def test_create_parallel_wrapper(self):
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
        
        with patch('morph.utils.distributed.DataParallelWrapper', return_value="data_parallel_model"):
            wrapped = create_parallel_wrapper(model, config)
            assert wrapped == "data_parallel_model"
        
        # Test expert parallel mode
        config.parallel_strategy = "expert_parallel"
        
        with patch('morph.utils.distributed.ExpertParallelWrapper', return_value="expert_parallel_model"):
            wrapped = create_parallel_wrapper(model, config)
            assert wrapped == "expert_parallel_model"
        
        # Test unknown strategy
        config.parallel_strategy = "unknown_strategy"
        wrapped = create_parallel_wrapper(model, config)
        assert wrapped is model  # Should return the original model
    
    def test_setup_distributed_environment(self):
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
