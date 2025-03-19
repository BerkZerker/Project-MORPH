import torch
import pytest


def test_prune_dormant_experts(model, model_config):
    """Test that dormant experts get pruned."""
    # Number of experts before
    num_experts_before = len(model.experts)
    
    # Mark one expert as dormant
    model.experts[2].last_activated = 0
    model.knowledge_graph.graph.nodes[2]['last_activated'] = 0
    model.knowledge_graph.graph.nodes[2]['activation_count'] = 10  # Below threshold
    
    # Set the step count to trigger dormant detection
    model.step_count = model_config.dormant_steps_threshold + 100
    
    # Call pruning directly
    pruned = model._prune_dormant_experts()
    
    # Check that an expert was pruned
    assert pruned, "Pruning should have occurred"
    assert len(model.experts) < num_experts_before, "Number of experts should decrease"
