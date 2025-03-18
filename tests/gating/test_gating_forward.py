import torch
import pytest
from src.core.gating import GatingNetwork


def test_gating_forward():
    """Test that a gating network forward pass works correctly."""
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gating = GatingNetwork(input_size=10, num_experts=4, k=2).to(device)
    
    # Create a batch of inputs on the appropriate device
    inputs = torch.randn(32, 10, device=device)
    
    # Forward pass
    routing_weights, expert_indices, uncertainty = gating(inputs, training=True)
    
    # Check output shapes
    assert routing_weights.shape == (32, 2)  # [batch_size, k]
    assert expert_indices.shape == (32, 2)  # [batch_size, k]
    assert uncertainty.dim() == 0  # scalar
    
    # Check device consistency
    assert routing_weights.device == device
    assert expert_indices.device == device
    assert uncertainty.device == device
    
    # Check routing weights sum to 1
    assert torch.allclose(routing_weights.sum(dim=1), torch.ones(32, device=device), atol=1e-6)
    
    # Check expert indices are in valid range
    assert torch.all(expert_indices >= 0)
    assert torch.all(expert_indices < 4)
    
    # Check uncertainty is between 0 and 1
    assert 0 <= uncertainty.item() <= 1
