import pytest
import torch
import torch.nn as nn
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.gpu_utils import (
    detect_available_gpus,
    get_gpu_memory_info,
    select_best_gpu,
    detect_and_select_gpu,
    estimate_max_batch_size,
    distribute_experts_across_gpus,
    set_gpu_memory_fraction,
    setup_gpu_environment,
    get_optimal_worker_count
)


# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestGpuUtils:
    
    def test_detect_available_gpus(self):
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
    
    def test_get_gpu_memory_info(self):
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
    
    def test_select_best_gpu(self):
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
    
    def test_detect_and_select_gpu(self):
        """Test GPU mode and device selection."""
        # Test when CUDA is not available
        with patch('torch.cuda.is_available', return_value=False):
            mode, devices = detect_and_select_gpu()
            assert mode == "cpu"
            assert len(devices) == 1
            assert devices[0].type == "cpu"
        
        # Test with single GPU
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                mode, devices = detect_and_select_gpu()
                assert mode == "single_gpu"
                assert len(devices) == 1
                assert devices[0].type == "cuda"
                assert devices[0].index == 0
        
        # Test with multiple GPUs
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=2):
                mode, devices = detect_and_select_gpu()
                assert mode == "multi_gpu"
                assert len(devices) == 2
                assert all(d.type == "cuda" for d in devices)
                assert devices[0].index == 0
                assert devices[1].index == 1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_estimate_max_batch_size_real(self):
        """Test batch size estimation with real GPU if available."""
        # Only run this test if CUDA is actually available
        model = SimpleModel()
        input_shape = (10,)  # Match model input size
        device = torch.device("cuda:0")
        
        # Move model to GPU
        model = model.to(device)
        
        # Estimate batch size
        batch_size = estimate_max_batch_size(model, input_shape, device)
        
        # Just check that we get a reasonable positive number
        assert batch_size > 0
        assert isinstance(batch_size, int)
    
    def test_estimate_max_batch_size_mock(self):
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
    
    def test_distribute_experts_across_gpus(self):
        """Test expert distribution across GPUs."""
        # Test with CPU only
        cpu_device = torch.device("cpu")
        expert_map = distribute_experts_across_gpus(5, [cpu_device])
        assert len(expert_map) == 5
        assert all(device.type == "cpu" for device in expert_map.values())
        
        # Test with multiple GPUs
        cuda0 = torch.device("cuda:0")
        cuda1 = torch.device("cuda:1")
        expert_map = distribute_experts_across_gpus(5, [cuda0, cuda1])
        
        # Should distribute 3 experts to cuda:0 and 2 to cuda:1
        assert len(expert_map) == 5
        assert sum(1 for device in expert_map.values() if device == cuda0) == 3
        assert sum(1 for device in expert_map.values() if device == cuda1) == 2
        
        # Test with mixed devices
        expert_map = distribute_experts_across_gpus(4, [cpu_device, cuda0])
        assert len(expert_map) == 4
        # Should prefer CUDA devices
        assert all(device.type == "cuda" for device in expert_map.values())
    
    def test_set_gpu_memory_fraction(self):
        """Test setting GPU memory fraction."""
        # This is mostly a coverage test since the function may not do anything
        # depending on PyTorch version
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise an error when CUDA is not available
            set_gpu_memory_fraction(0.5)
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=2):
                with patch('torch.cuda.set_per_process_memory_fraction') as mock_set:
                    set_gpu_memory_fraction(0.5)
                    # Should be called once per GPU
                    assert mock_set.call_count == 2
    
    def test_setup_gpu_environment(self):
        """Test GPU environment setup."""
        # This is mostly a coverage test
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise an error when CUDA is not available
            setup_gpu_environment()
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.backends.cuda.matmul.allow_tf32', True):
                with patch('torch.backends.cudnn.allow_tf32', True):
                    with patch('torch.backends.cudnn.benchmark', True):
                        with patch('src.utils.gpu_utils.set_gpu_memory_fraction') as mock_set:
                            setup_gpu_environment()
                            # Should call set_gpu_memory_fraction
                            mock_set.assert_called_once()
    
    def test_get_optimal_worker_count(self):
        """Test worker count determination."""
        # Test with mocked CPU count
        with patch('multiprocessing.cpu_count', return_value=8):
            with patch('torch.cuda.is_available', return_value=False):
                # CPU only mode
                workers = get_optimal_worker_count()
                assert workers == 4  # Default for CPU
            
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.device_count', return_value=2):
                    # 2 GPUs, should use 8 workers (4 per GPU)
                    workers = get_optimal_worker_count()
                    assert workers == 8
                
                with patch('torch.cuda.device_count', return_value=4):
                    # 4 GPUs, but capped by CPU count
                    workers = get_optimal_worker_count()
                    assert workers == 8  # Capped at CPU count
