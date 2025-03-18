import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph
from src.utils.testing.decorators import visualize_test, capture_test_state


@visualize_test
def test_get_similar_experts():
    """Test finding similar experts in the knowledge graph."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add several experts
    for i in range(5):
        kg.add_expert(i)
    
    # Add similarity edges
    kg.add_edge(0, 1, weight=0.9, relation_type='similarity')
    kg.add_edge(0, 2, weight=0.3, relation_type='similarity')
    kg.add_edge(0, 3, weight=0.8, relation_type='similarity')
    kg.add_edge(0, 4, weight=0.5, relation_type='similarity')
    
    # Find similar experts with high threshold and visualization
    with capture_test_state(kg, "Get Similar Experts"):
        similar = kg.get_similar_experts(0, threshold=0.7)
    
    # Should return experts 1 and 3
    assert len(similar) == 2
    assert (1, 0.9) in similar
    assert (3, 0.8) in similar
    
    # Check sorting (highest similarity first)
    assert similar[0][1] >= similar[1][1]
    
    # Test with non-existent expert
    assert kg.get_similar_experts(999) == []
