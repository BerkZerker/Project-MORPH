import torch
from src.core.expert import Expert




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
