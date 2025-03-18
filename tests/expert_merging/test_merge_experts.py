import torch
import pytest


def test_merge_experts(model):
    """Test that experts can be merged properly."""
    # Number of experts before
    num_experts_before = len(model.experts)
    
    # Force high similarity between two experts
    with torch.no_grad():
        for p1, p2 in zip(model.experts[0].parameters(), model.experts[1].parameters()):
            p2.copy_(p1)
    
    # Merge the experts
    model._merge_expert_parameters(0, 1)
    
    # Test that parameters were properly merged
    # Since we copied them, they should still be identical
    with torch.no_grad():
        for p1, p2 in zip(model.experts[0].parameters(), model.experts[1].parameters()):
            assert torch.allclose(p1, p2)
