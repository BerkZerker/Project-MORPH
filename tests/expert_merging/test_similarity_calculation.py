import torch
import pytest


def test_similarity_calculation(model):
    """Test that expert similarity calculation works properly."""
    # Clone an expert to get high similarity
    expert1 = model.experts[0]
    expert2 = expert1.clone()
    
    # Copy parameters exactly
    with torch.no_grad():
        for p1, p2 in zip(expert1.parameters(), expert2.parameters()):
            p2.copy_(p1)
            
    # Calculate similarity
    similarity = expert1.get_parameter_similarity(expert2)
    assert similarity > 0.99, "Identical experts should have similarity near 1.0"
    
    # Create a very different expert
    expert3 = expert1.clone()
    with torch.no_grad():
        for p in expert3.parameters():
            p.data = torch.randn_like(p)
            
    # Calculate similarity
    similarity = expert1.get_parameter_similarity(expert3)
    assert similarity < 0.9, "Different experts should have lower similarity"
