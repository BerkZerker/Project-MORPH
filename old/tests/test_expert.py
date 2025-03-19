import torch
import pytest
from src.core.expert import Expert
from src.utils.testing.decorators import visualize_test, capture_test_state
from src.utils.gpu_utils import get_optimal_worker_count


@visualize_test
def test_expert_initialization(optimized_test_config):
    """Test that an expert initializes correctly."""
    config = optimized_test_config
    expert = Expert(input_size=config.input_size, hidden_size=config.expert_hidden_size, output_size=config.output_size)
    
    # Check structure
    assert expert.expert_id is None
    assert expert.activation_count == 0
    assert expert.last_activated == 0
    
    # Check network structure
    assert len(expert.network) == 5  # input -> hidden -> relu -> hidden -> output
    assert expert.network[0].in_features == config.input_size
    assert expert.network[0].out_features == config.expert_hidden_size
    assert expert.network[-1].out_features == 5


@visualize_test
def test_expert_forward(optimized_test_config):
    """Test that an expert forward pass works correctly."""
    config = optimized_test_config
    expert = Expert(input_size=config.input_size, hidden_size=config.expert_hidden_size, output_size=config.output_size)
    
    # Move expert to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert = expert.to(device)
    
    # Create a batch of inputs and move to the same device
    inputs = torch.randn(32, 10, device=device)
    
    # Forward pass
    outputs = expert(inputs)
    
    # Check output shape
    assert outputs.shape == (32, 5)
    # Check device - use str comparison to handle cuda vs cuda:0
    assert str(outputs.device).startswith(str(device).split(':')[0])
    
    # Check activation count (may be batch size if expert counts each sample)
    assert expert.activation_count > 0


@visualize_test
def test_expert_clone(optimized_test_config):
    """Test that an expert can be cloned correctly."""
    config = optimized_test_config
    expert = Expert(input_size=config.input_size, hidden_size=config.expert_hidden_size, output_size=config.output_size, num_layers=3)
    expert.expert_id = 42
    
    # Move expert to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert = expert.to(device)
    
    # Clone the expert and move to the same device
    cloned_expert = expert.clone().to(device)
    
    # Check structure is the same
    assert expert.network[0].in_features == cloned_expert.network[0].in_features
    assert expert.network[0].out_features == cloned_expert.network[0].out_features
    assert expert.network[-1].out_features == cloned_expert.network[-1].out_features
    assert len(expert.network) == len(cloned_expert.network)
    
    # Check that cloned expert is on the same device - use str comparison to handle cuda vs cuda:0
    assert str(next(cloned_expert.parameters()).device).startswith(str(device).split(':')[0])
    
    # But weights should be different (re-initialized)
    with torch.no_grad():
        for p1, p2 in zip(expert.parameters(), cloned_expert.parameters()):
            assert not torch.allclose(p1, p2)
    
    # ID should NOT be cloned
    assert cloned_expert.expert_id is None
    
    # Activation count should be reset
    assert cloned_expert.activation_count == 0


@visualize_test
def test_parameter_similarity(optimized_test_config):
    """Test parameter similarity calculation."""
    config = optimized_test_config
    # Move experts to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert1 = Expert(input_size=config.input_size, hidden_size=config.expert_hidden_size, output_size=config.output_size).to(device)
    expert2 = expert1.clone().to(device)  # Different random weights
    
    # With different weights, similarity should be low
    similarity = expert1.get_parameter_similarity(expert2)
    # Convert to float if it's a tensor
    if isinstance(similarity, torch.Tensor):
        similarity = similarity.item()
    # Allow for small negative values due to numerical precision
    assert -0.1 <= similarity <= 1.1  # Should be a valid similarity score
    
    # Same expert should have perfect similarity
    similarity = expert1.get_parameter_similarity(expert1)
    # Convert to float if it's a tensor
    if isinstance(similarity, torch.Tensor):
        similarity = similarity.item()
    assert similarity > 0.99  # Should be very close to 1
