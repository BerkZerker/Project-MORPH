import torch
import pytest
from src.config import MorphConfig
from src.core.model import MorphModel

from src.utils.gpu_utils import get_optimal_worker_count



def test_sleep_cycle_initialization():
    """Test that sleep cycle metrics are properly initialized."""
    config = MorphConfig(
        input_size=10,
        expert_hidden_size=20,
        output_size=5,
        num_initial_experts=3,
        
        # Sleep settings
        enable_sleep=True,
        sleep_cycle_frequency=100,
        enable_adaptive_sleep=True,
        
        # Use GPU if available, otherwise CPU
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Enable mixed precision if CUDA is available
        enable_mixed_precision=torch.cuda.is_available(),
        # Optimize data loading
        num_workers=get_optimal_worker_count(),
        pin_memory=torch.cuda.is_available()
    )
    
    model = MorphModel(config)
    
    # Check sleep cycle counters
    assert model.sleep_cycles_completed == 0
    assert model.next_sleep_step == config.sleep_cycle_frequency
    assert model.adaptive_sleep_frequency == config.sleep_cycle_frequency
