import torch
import pytest
from src.core.gating import GatingNetwork


def test_noisy_routing():
    """Test that noisy routing adds noise during training."""
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gating = GatingNetwork(input_size=10, num_experts=4, k=2, routing_type="noisy_top_k").to(device)
    
    # Create a batch of inputs on the appropriate device
    inputs = torch.randn(32, 10, device=device)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Forward pass with training=True
    routing_weights1, expert_indices1, _ = gating(inputs, training=True)
    
    # Set the same seed again
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Forward pass with training=False
    routing_weights2, expert_indices2, _ = gating(inputs, training=False)
    
    # The results should be different due to noise
    assert not torch.allclose(routing_weights1, routing_weights2)
