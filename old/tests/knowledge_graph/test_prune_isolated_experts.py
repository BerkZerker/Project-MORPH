import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph
from src.utils.testing.decorators import visualize_test, capture_test_state


@visualize_test
def test_prune_isolated_experts():
    """Test identifying isolated experts in the knowledge graph."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add several experts
    for i in range(4):
        kg.add_expert(i)
    
    # Connect some experts
    kg.add_edge(0, 1, weight=0.8)
    kg.add_edge(1, 2, weight=0.6)
    # Expert 3 remains isolated
    
    # Find isolated experts with visualization
    with capture_test_state(kg, "Prune Isolated Experts"):
        isolated = kg.prune_isolated_experts()
    
    # Only expert 3 should be isolated
    assert len(isolated) == 1
    assert 3 in isolated
