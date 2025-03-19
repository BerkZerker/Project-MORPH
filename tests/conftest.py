"""
Pytest configuration for MORPH tests.
"""

import pytest
import torch

from src.config import MorphConfig
from src.utils.gpu_utils import get_optimal_worker_count

# Global test configuration fixture
@pytest.fixture(scope="session")
def optimized_test_config():
    """
    Create an optimized configuration for tests.
    
    This configuration uses smaller model sizes and reduced parameters
    to speed up test execution while maintaining functionality.
    """
    config = MorphConfig()
    
    # Enable test mode
    config.test_mode = True
    
    # Reduced model size
    config.input_size = 10  # Much smaller than default 784
    config.expert_hidden_size = 32  # Much smaller than default 256
    config.output_size = 5  # Smaller than default 10
    config.num_initial_experts = 2  # Fewer initial experts
    
    # Sleep cycle optimization
    config.sleep_cycle_frequency = 100  # Reduced from 1000
    config.memory_buffer_size = 200  # Reduced from 2000
    config.memory_replay_batch_size = 8  # Reduced from 32
    
    # Batch size optimization
    config.batch_size = 16  # Reduced from 64
    
    # Still use GPU if available (as requested)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.enable_mixed_precision = torch.cuda.is_available()
    config.num_workers = get_optimal_worker_count()
    config.pin_memory = torch.cuda.is_available()
    
    return config
