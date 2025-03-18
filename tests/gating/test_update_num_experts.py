import torch
import pytest
from src.core.gating import GatingNetwork


def test_update_num_experts():
    """Test updating the gating network for more experts."""
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gating = GatingNetwork(input_size=10, num_experts=4, k=2).to(device)
    
    # Update to more experts
    gating.update_num_experts(6)
    
    # Check attributes updated
    assert gating.num_experts == 6
    assert gating.k == 2  # k shouldn't change
    
    # Check network updated
    assert gating.router[-1].out_features == 6
    
    # Check we can still do a forward pass
    inputs = torch.randn(32, 10, device=device)
    routing_weights, expert_indices, uncertainty = gating(inputs)
    
    # Check device consistency
    assert routing_weights.device == device
    assert expert_indices.device == device
    
    # Expert indices should be in the new range
    assert torch.all(expert_indices >= 0)
    assert torch.all(expert_indices < 6)
