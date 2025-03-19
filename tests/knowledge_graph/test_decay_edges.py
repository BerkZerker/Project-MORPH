import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph




def test_decay_edges():
    """Test edge decay functionality."""
    config = MorphConfig(
        knowledge_edge_decay=0.5,  # Aggressive decay for testing
        knowledge_edge_min=0.2
    )
    kg = KnowledgeGraph(config)
    
    # Add experts and edges
    kg.add_expert(0)
    kg.add_expert(1)
    kg.add_expert(2)
    
    kg.add_edge(0, 1, weight=1.0)
    kg.add_edge(0, 2, weight=0.3)  # Just above min threshold
    
    # Apply decay
    kg.decay_edges()
    
    # First edge should be decayed but still exist
    assert kg.graph.has_edge(0, 1)
    assert kg.graph.get_edge_data(0, 1)['weight'] == 0.5  # 1.0 * 0.5
    
    # Second edge should be decayed below threshold and removed
    assert not kg.graph.has_edge(0, 2)  # 0.3 * 0.5 = 0.15 < 0.2
