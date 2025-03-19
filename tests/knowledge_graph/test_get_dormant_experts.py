import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph




def test_get_dormant_experts():
    """Test identifying dormant experts."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add experts with different activation patterns
    for i in range(4):
        kg.add_expert(i)
        
    # Update activation history
    kg.update_expert_activation(0, step=100)  # Recently activated
    kg.update_expert_activation(1, step=10)   # Dormant
    kg.update_expert_activation(2, step=20)   # Dormant
    # Expert 3 never activated
    
    # Make expert 0 have many activations (not dormant despite inactivity)
    kg.graph.nodes[0]['activation_count'] = 200
    
    # Set expert 2 to have few activations (dormant due to both inactivity and few activations)
    kg.graph.nodes[2]['activation_count'] = 5
    
    # Get dormant experts
    current_step = 100
    dormant = kg.get_dormant_experts(
        current_step=current_step,
        dormancy_threshold=50,  # Inactive for 50+ steps
        min_activations=10      # Fewer than 10 activations
    )
    
    # Experts 1 and 2 should be dormant (inactive + few activations)
    # Expert 0 has too many activations to be dormant
    # Expert 3 was never activated so should be dormant
    assert 1 in dormant
    assert 2 in dormant
    assert 3 in dormant
    assert 0 not in dormant
