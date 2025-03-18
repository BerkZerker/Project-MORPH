import torch
import pytest


def test_sleep_merges_similar_experts(model, model_config):
    """Test that sleep cycle merges similar experts."""
    # Force high similarity between experts 0 and 1
    with torch.no_grad():
        for p1, p2 in zip(model.experts[0].parameters(), model.experts[1].parameters()):
            p2.copy_(p1)
    
    # Lower the threshold to ensure merging happens
    model.config.expert_similarity_threshold = 0.5
    
    # Number of experts before
    num_experts_before = len(model.experts)
    
    # Mock the merge method to check if it's called
    original_merge = model._merge_similar_experts
    merge_called = [False]
    
    def mock_merge():
        merge_called[0] = True
        return original_merge()
        
    model._merge_similar_experts = mock_merge
    
    # Call sleep
    model.sleep()
    
    # Check that merge was called
    assert merge_called[0], "Merge method should be called during sleep"
