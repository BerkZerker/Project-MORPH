import torch
import pytest
from src.core.gating import GatingNetwork


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
