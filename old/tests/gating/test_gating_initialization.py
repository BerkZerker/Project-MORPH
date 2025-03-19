import torch
import pytest
from src.core.gating import GatingNetwork


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
