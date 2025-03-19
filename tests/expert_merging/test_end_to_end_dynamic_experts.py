import torch
import pytest


def test_end_to_end_dynamic_experts(model, model_config):
    """Test the full dynamic expert lifecycle with merging and pruning."""
    initial_expert_count = len(model.experts)
    
    # Get the device from the model
    device = model.device
    
    # Process batches to trigger expert creation
    for i in range(10):
        batch = torch.randn(10, model_config.input_size, device=device)
        model(batch, training=True)
        
    # Force similar experts to trigger merging
    with torch.no_grad():
        for p1, p2 in zip(model.experts[0].parameters(), model.experts[1].parameters()):
            p2.copy_(p1)
            
    # Lower the similarity threshold to ensure merging
    model.config.expert_similarity_threshold = 0.5
            
    # Process more batches to trigger sleep
    for i in range(model_config.sleep_cycle_frequency):
        batch = torch.randn(10, model_config.input_size, device=device)
        model(batch, training=True)
        
    # Check that experts were dynamically managed
    assert len(model.experts) != initial_expert_count, "Expert count should change after dynamic lifecycle"
