import torch
import pytest
from src.config import MorphConfig
from src.core.model import MorphModel

from src.utils.gpu_utils import get_optimal_worker_count



def test_memory_replay():
    """Test that memory replay works as expected."""
    config = MorphConfig(
        input_size=10,
        expert_hidden_size=20,
        output_size=5,
        num_initial_experts=2,
        expert_k=1,
        
        # Sleep settings
        enable_sleep=True,
        sleep_cycle_frequency=100,
        
        # Use GPU if available, otherwise CPU
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Enable mixed precision if CUDA is available
        enable_mixed_precision=torch.cuda.is_available(),
        # Optimize data loading
        num_workers=get_optimal_worker_count(),
        pin_memory=torch.cuda.is_available()
    )
    
    model = MorphModel(config)
    
    # Create some fake activations
    for expert_idx in range(len(model.experts)):
        for _ in range(10):
            # Create inputs on the same device as the model
            inputs = torch.randn(2, config.input_size).to(model.device)
            outputs = model.experts[expert_idx](inputs)
            
            # Store CPU tensors in the buffer to avoid device issues
            model.activation_buffer.append({
                'expert_idx': expert_idx,
                'inputs': inputs.cpu(),
                'outputs': outputs.cpu(),
                'routing_weight': 1.0,
                'step': 0,
                'batch_size': inputs.size(0),
                'input_features': torch.mean(inputs, dim=0).cpu(),
                'uncertainty': 0.1
            })
    
    # Verify activation buffer has been populated
    assert len(model.activation_buffer) == 20
    
    # Perform memory replay
    result = model._perform_memory_replay()
    
    # Verify memory replay worked
    assert result is True
    assert len(model.activation_buffer) == 0  # Buffer should be cleared
