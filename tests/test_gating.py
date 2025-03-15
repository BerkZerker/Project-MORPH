import torch
import pytest
from morph.core.gating import GatingNetwork


def test_gating_initialization():
    """Test that a gating network initializes correctly."""
    gating = GatingNetwork(input_size=10, num_experts=4, k=2)
    
    # Check attributes
    assert gating.input_size == 10
    assert gating.num_experts == 4
    assert gating.k == 2
    assert gating.routing_type == "top_k"
    
    # Check network structure
    assert len(gating.router) == 3  # linear -> relu -> linear
    assert gating.router[0].in_features == 10
    assert gating.router[-1].out_features == 4


def test_gating_forward():
    """Test that a gating network forward pass works correctly."""
    gating = GatingNetwork(input_size=10, num_experts=4, k=2)
    
    # Create a batch of inputs
    inputs = torch.randn(32, 10)
    
    # Forward pass
    routing_weights, expert_indices, uncertainty = gating(inputs, training=True)
    
    # Check output shapes
    assert routing_weights.shape == (32, 2)  # [batch_size, k]
    assert expert_indices.shape == (32, 2)  # [batch_size, k]
    assert uncertainty.dim() == 0  # scalar
    
    # Check routing weights sum to 1
    assert torch.allclose(routing_weights.sum(dim=1), torch.ones(32), atol=1e-6)
    
    # Check expert indices are in valid range
    assert torch.all(expert_indices >= 0)
    assert torch.all(expert_indices < 4)
    
    # Check uncertainty is between 0 and 1
    assert 0 <= uncertainty.item() <= 1


def test_noisy_routing():
    """Test that noisy routing adds noise during training."""
    gating = GatingNetwork(input_size=10, num_experts=4, k=2, routing_type="noisy_top_k")
    
    # Create a batch of inputs
    inputs = torch.randn(32, 10)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Forward pass with training=True
    routing_weights1, expert_indices1, _ = gating(inputs, training=True)
    
    # Set the same seed again
    torch.manual_seed(42)
    
    # Forward pass with training=False
    routing_weights2, expert_indices2, _ = gating(inputs, training=False)
    
    # The results should be different due to noise
    assert not torch.allclose(routing_weights1, routing_weights2)


def test_should_create_new_expert():
    """Test the expert creation decision logic."""
    gating = GatingNetwork(input_size=10, num_experts=4)
    gating.uncertainty_threshold = 0.5
    
    # Uncertainty below threshold
    assert not gating.should_create_new_expert(0.4)
    
    # Uncertainty at threshold
    assert not gating.should_create_new_expert(0.5)
    
    # Uncertainty above threshold
    assert gating.should_create_new_expert(0.6)


def test_update_num_experts():
    """Test updating the gating network for more experts."""
    gating = GatingNetwork(input_size=10, num_experts=4, k=2)
    
    # Update to more experts
    gating.update_num_experts(6)
    
    # Check attributes updated
    assert gating.num_experts == 6
    assert gating.k == 2  # k shouldn't change
    
    # Check network updated
    assert gating.router[-1].out_features == 6
    
    # Check we can still do a forward pass
    inputs = torch.randn(32, 10)
    routing_weights, expert_indices, uncertainty = gating(inputs)
    
    # Expert indices should be in the new range
    assert torch.all(expert_indices >= 0)
    assert torch.all(expert_indices < 6)
