import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph
from src.utils.testing.decorators import visualize_test, capture_test_state


@visualize_test
def test_merge_expert_connections():
    """Test merging expert connections when removing an expert."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add experts
    for i in range(5):
        kg.add_expert(i)
    
    # Add connections to expert 0 (to be removed)
    kg.add_edge(0, 1, weight=0.8, relation_type='similarity')
    kg.add_edge(0, 2, weight=0.6, relation_type='dependency')
    kg.add_edge(0, 3, weight=0.9, relation_type='specialization')
    # No connection to expert 4
    
    # Merge connections from expert 0 to experts 1 and 3 with visualization
    with capture_test_state(kg, "Merge Expert Connections"):
        kg.merge_expert_connections(0, target_ids=[1, 3])
    
    # Check that expert 0's connections were transferred
    # Expert 1 should now be connected to experts 2 and 3
    assert kg.graph.has_edge(1, 2)
    assert kg.graph.get_edge_data(1, 2)['relation_type'] == 'dependency'
    assert kg.graph.get_edge_data(1, 2)['weight'] < 0.6  # Should be reduced
    
    # Expert 3 should now be connected to expert 2
    assert kg.graph.has_edge(3, 2)
    assert kg.graph.get_edge_data(3, 2)['relation_type'] == 'dependency'
    assert kg.graph.get_edge_data(3, 2)['weight'] < 0.6  # Should be reduced
    
    # No connection should be created to expert 4
    assert not kg.graph.has_edge(1, 4)
    assert not kg.graph.has_edge(3, 4)
    
    # Test with non-existent source expert
    kg.merge_expert_connections(999, [1, 3])  # Should not raise error
