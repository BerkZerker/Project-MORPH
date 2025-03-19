import pytest
import torch

from src.utils.gpu_utils import distribute_experts_across_gpus


def test_distribute_experts_across_gpus():
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
