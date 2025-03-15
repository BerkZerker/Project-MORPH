import torch
import pytest
from morph.core.expert import Expert


def test_expert_initialization():
    """Test that an expert initializes correctly."""
    expert = Expert(input_size=10, hidden_size=20, output_size=5)
    
    # Check structure
    assert expert.expert_id is None
    assert expert.activation_count == 0
    assert expert.last_activated == 0
    
    # Check network structure
    assert len(expert.network) == 5  # input -> hidden -> relu -> hidden -> output
    assert expert.network[0].in_features == 10
    assert expert.network[0].out_features == 20
    assert expert.network[-1].out_features == 5


def test_expert_forward():
    """Test that an expert forward pass works correctly."""
    expert = Expert(input_size=10, hidden_size=20, output_size=5)
    
    # Create a batch of inputs
    inputs = torch.randn(32, 10)
    
    # Forward pass
    outputs = expert(inputs)
    
    # Check output shape
    assert outputs.shape == (32, 5)
    
    # Check activation count
    assert expert.activation_count == 1


def test_expert_clone():
    """Test that an expert can be cloned correctly."""
    expert = Expert(input_size=10, hidden_size=20, output_size=5, num_layers=3)
    expert.expert_id = 42
    
    # Clone the expert
    cloned_expert = expert.clone()
    
    # Check structure is the same
    assert expert.network[0].in_features == cloned_expert.network[0].in_features
    assert expert.network[0].out_features == cloned_expert.network[0].out_features
    assert expert.network[-1].out_features == cloned_expert.network[-1].out_features
    assert len(expert.network) == len(cloned_expert.network)
    
    # But weights should be different (re-initialized)
    with torch.no_grad():
        for p1, p2 in zip(expert.parameters(), cloned_expert.parameters()):
            assert not torch.allclose(p1, p2)
    
    # ID should NOT be cloned
    assert cloned_expert.expert_id is None
    
    # Activation count should be reset
    assert cloned_expert.activation_count == 0


def test_parameter_similarity():
    """Test parameter similarity calculation."""
    expert1 = Expert(input_size=10, hidden_size=20, output_size=5)
    expert2 = expert1.clone()  # Different random weights
    
    # With different weights, similarity should be low
    similarity = expert1.get_parameter_similarity(expert2)
    assert 0 <= similarity <= 1  # Should be a valid similarity score
    
    # Same expert should have perfect similarity
    similarity = expert1.get_parameter_similarity(expert1)
    assert similarity > 0.99  # Should be very close to 1
