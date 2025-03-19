import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph




def test_rebuild_graph():
    """Test rebuilding the knowledge graph after expert count changes."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add experts
    for i in range(4):
        kg.add_expert(i)
        kg.graph.nodes[i]['custom_field'] = f"expert_{i}"
    
    # Add edges
    kg.add_edge(0, 1, weight=0.8)
    kg.add_edge(1, 2, weight=0.6)
    kg.add_edge(2, 3, weight=0.7)
    
    # Track original state
    original_node_count = len(kg.graph.nodes)
    original_edge_count = len(kg.graph.edges)
    
    # Rebuild with fewer experts (removing expert 3)
    kg.rebuild_graph(expert_count=3)
    
    # Check node count changed
    assert len(kg.graph.nodes) == 3
    assert original_node_count == 4
    
    # Check edges were preserved for remaining experts
    assert kg.graph.has_edge(0, 1)
    assert kg.graph.has_edge(1, 2)
    assert not kg.graph.has_edge(2, 3)  # This edge should be gone
    
    # Check node attributes were preserved
    assert kg.graph.nodes[0]['custom_field'] == "expert_0"
    
    # Check expert concepts mapping was updated
    assert len(kg.expert_concepts) == 3
    assert 3 not in kg.expert_concepts
