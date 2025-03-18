import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph
from src.utils.testing.decorators import visualize_test, capture_test_state


@visualize_test
def test_add_edge():
    """Test adding edges between experts in the knowledge graph."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add two experts
    kg.add_expert(0)
    kg.add_expert(1)
    
    # Add an edge between them with visualization
    with capture_test_state(kg, "Add Edge"):
        kg.add_edge(
            expert1_id=0,
            expert2_id=1,
            weight=0.75,
            relation_type='similarity'
        )
    
    # Check edge was added
    assert kg.graph.has_edge(0, 1)
    edge_data = kg.graph.get_edge_data(0, 1)
    assert edge_data['weight'] == 0.75
    assert edge_data['relation_type'] == 'similarity'
    
    # Test with invalid relation type
    kg.add_edge(0, 1, relation_type='invalid_type')
    # Should default to the valid relation type
    assert kg.graph.get_edge_data(0, 1)['relation_type'] in config.knowledge_relation_types
