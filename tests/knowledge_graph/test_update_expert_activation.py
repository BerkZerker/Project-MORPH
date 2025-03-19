import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph




def test_update_expert_activation():
    """Test updating expert activation information."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add an expert
    kg.add_expert(0)
    assert kg.graph.nodes[0]['activation_count'] == 0
    assert kg.graph.nodes[0]['last_activated'] == 0
    
    # Update activation
    kg.update_expert_activation(0, step=42)
    assert kg.graph.nodes[0]['activation_count'] == 1
    assert kg.graph.nodes[0]['last_activated'] == 42
    
    # Update again
    kg.update_expert_activation(0, step=100)
    assert kg.graph.nodes[0]['activation_count'] == 2
    assert kg.graph.nodes[0]['last_activated'] == 100
    
    # Test with non-existent expert
    kg.update_expert_activation(999, step=1)  # Should not raise error
