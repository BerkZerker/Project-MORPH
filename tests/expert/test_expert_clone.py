import torch
from src.core.expert import Expert
from src.utils.testing.decorators import visualize_test, capture_test_state


@visualize_test
def test_expert_clone():
    """Test that an expert can be cloned correctly."""
    expert = Expert(input_size=10, hidden_size=20, output_size=5, num_layers=3)
    expert.expert_id = 42
    
    # Move expert to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert = expert.to(device)
    
    # Clone the expert with visualization
    with capture_test_state(expert, "Expert Cloning"):
        cloned_expert = expert.clone()
    
    # Check structure is the same
    assert expert.network[0].in_features == cloned_expert.network[0].in_features
    assert expert.network[0].out_features == cloned_expert.network[0].out_features
    assert expert.network[-1].out_features == cloned_expert.network[-1].out_features
    assert len(expert.network) == len(cloned_expert.network)
    
    # Check that cloned expert is on the same device
    assert next(cloned_expert.parameters()).device == device
    
    # But weights should be different (re-initialized)
    with torch.no_grad():
        for p1, p2 in zip(expert.parameters(), cloned_expert.parameters()):
            assert not torch.allclose(p1, p2)
    
    # ID should NOT be cloned
    assert cloned_expert.expert_id is None
    
    # Activation count should be reset
    assert cloned_expert.activation_count == 0
